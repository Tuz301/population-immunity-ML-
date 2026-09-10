"""Gradient-boosted reach model: the share of children the next round will reach.

This is the learned layer. It does not predict how many campaigns are needed; it
predicts the one quantity the epidemiological engine cannot derive from first
principles, which is how well a team will do in this place next time.

Two requirements pull against each other here.

*Domain constraints.* The model must never forecast better reach because
insecurity rose or because the settlement was missed more often. Those signs are
known, and a planning model that gets them backwards is worse than no model.
LightGBM enforces them through monotone constraints.

*Honest intervals.* A point forecast of reach produces a point forecast of the
round count, which hides the difference between "three rounds, confidently" and
"three rounds if everything goes right, seven if it does not".

LightGBM will not accept monotone constraints under a quantile objective, so the
two are separated. A constrained squared-error model gives the centre. A second
model gives the conditional spread. The shape of the standardised residual is
then taken from data the centre model never saw, and quantiles are rebuilt from
the three. This is the split-conformal construction, and on panels this size it
calibrates better than a set of independently fitted quantile regressions, which
tend to cross and to be over-confident where data are thin.

Validation is forward in time. Random k-fold on a panel like this scores a model
on rounds it has already seen in neighbouring settlements and reports a skill the
field will never see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, MONOTONE_CONSTRAINTS

#: Quantiles reported. The spread between them becomes the width of the
#: round-count interval.
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

#: Floor on the conditional spread, in reach points. Stops a confident-looking
#: region of the feature space from producing a zero-width interval.
MIN_SCALE = 0.02



@dataclass
class ReachModelReport:
    """How well the reach model performed, and where it did not.

    Attributes:
        n_train: Rows used for fitting.
        n_calibration: Rows used to learn the residual shape.
        n_validation: Rows used for out-of-sample scoring.
        mae: Mean absolute error of the central forecast, in reach points.
        rmse: Root mean squared error of the central forecast.
        pinball_loss: Mean pinball loss across the reported quantiles.
        skill_vs_persistence: Share of the mean absolute error of the
            last-round-repeats baseline that this model removes. Zero means the
            model adds nothing over assuming next round looks like last round.
        interval_coverage: Observed coverage of each nominal prediction interval.
            A calibrated model puts the truth inside its 80% interval 80% of the
            time. Anything else is stated rather than smoothed.
        feature_importance: Gain-based importance in the centre model.
        notes: Anything a reader must know before using the numbers.
    """

    n_train: int
    n_calibration: int
    n_validation: int
    mae: float
    rmse: float
    pinball_loss: float
    skill_vs_persistence: float
    interval_coverage: dict[str, float]
    feature_importance: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def calibration_verdict(self, tolerance: float = 0.07) -> str:
        """One line on whether the intervals can be read as stated.

        Args:
            tolerance: Allowed gap between nominal and observed coverage.
        """
        failures = [
            f"{name} nominal, {value:.0%} observed"
            for name, value in self.interval_coverage.items()
            if abs(value - float(name.rstrip("%")) / 100.0) > tolerance
        ]
        if not failures:
            return (
                "Reach intervals are calibrated within tolerance, so the round-count "
                "intervals derived from them may be read as stated."
            )
        return (
            "Reach intervals are NOT calibrated within tolerance, so round-count intervals "
            "are indicative only. Failing levels: " + "; ".join(failures)
        )


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Mean pinball loss at one quantile. Lower is better."""
    delta = y_true - y_pred
    return float(np.mean(np.maximum(quantile * delta, (quantile - 1.0) * delta)))


class ReachModel:
    """Forecast of per-round reach, as a distribution rather than a number.

    Example:
        >>> model = ReachModel()  # doctest: +SKIP
        >>> report = model.fit(features, validation_rounds=2)  # doctest: +SKIP
        >>> print(report.calibration_verdict())  # doctest: +SKIP
        >>> quantiles = model.predict_quantiles(latest_features)  # doctest: +SKIP
    """

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        *,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 30,
        random_state: int = 42,
    ) -> None:
        if not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError("Every quantile must lie strictly between 0 and 1.")
        self.quantiles = tuple(sorted(quantiles))
        self.params: dict[str, Any] = {
            "objective": "regression",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "verbose": -1,
        }
        self.centre: lgb.LGBMRegressor | None = None
        self.scale: lgb.LGBMRegressor | None = None
        self.residual_quantiles: np.ndarray | None = None
        self.design_columns: list[str] = []
        self.report: ReachModelReport | None = None

    # -- design matrix -----------------------------------------------------

    def _design(self, features: pd.DataFrame, *, fitting: bool = False) -> pd.DataFrame:
        """Build the model matrix.

        Categorical columns are one-hot encoded rather than passed to LightGBM as
        native categoricals, because LightGBM refuses monotone constraints on any
        model holding a native categorical.
        """
        design = features.reindex(columns=FEATURE_COLUMNS).copy()
        for column in CATEGORICAL_COLUMNS:
            if column in features.columns:
                dummies = pd.get_dummies(features[column].astype(str), prefix=column, dtype=float)
                design = pd.concat([design, dummies], axis=1)
        if fitting:
            self.design_columns = list(design.columns)
        elif self.design_columns:
            design = design.reindex(columns=self.design_columns, fill_value=0.0)
        return design.astype(float)

    def _constraints(self, n_columns: int) -> list[int]:
        """Monotone constraint per design column; zero for one-hot indicators."""
        base = [MONOTONE_CONSTRAINTS.get(c, 0) for c in FEATURE_COLUMNS]
        return base + [0] * (n_columns - len(base))

    @staticmethod
    def _usable(features: pd.DataFrame) -> pd.Series:
        """Rows the model can learn from: a real target and a real denominator."""
        return features["reach"].notna() & (features["target_pop"] > 0)

    # -- fitting -----------------------------------------------------------

    def fit(self, features: pd.DataFrame, *, validation_rounds: int = 2) -> ReachModelReport:
        """Fit the model and score it on rounds it has never seen.

        The panel is cut twice, always forward in time. The last
        ``validation_rounds`` rounds are the holdout. Inside what remains, the
        last ``validation_rounds`` rounds are the calibration slice, and the rest
        fits the centre and scale models. The centre model is then refitted on
        everything before the holdout, so no data is wasted.

        The calibration slice is the same depth as the holdout on purpose. A
        residual measured one round ahead is smaller than one measured three
        rounds ahead, so calibrating at one horizon and deploying at another
        produces intervals that are too narrow exactly when they matter. Matching
        the depths makes the stated coverage mean what it says.

        Args:
            features: Output of ``features.build_features``.
            validation_rounds: Trailing rounds held out for scoring.

        Returns:
            A ``ReachModelReport``.

        Raises:
            ValueError: If too few usable rows or rounds remain.
        """
        usable = features[self._usable(features)].copy()
        if len(usable) < 50:
            raise ValueError(
                f"Only {len(usable)} usable rows. The reach model needs at least 50. "
                "Fall back to the programme reach prior instead of fitting."
            )

        rounds = np.sort(usable["round_index"].unique())
        if len(rounds) < 2 * validation_rounds + 2:
            raise ValueError(
                f"Panel has {len(rounds)} rounds; needs at least {2 * validation_rounds + 2} "
                f"to hold out {validation_rounds} for scoring and {validation_rounds} more "
                "for calibration at a matching horizon."
            )

        holdout_cut = rounds[-validation_rounds]
        train_all = usable[usable["round_index"] < holdout_cut]
        valid = usable[usable["round_index"] >= holdout_cut]

        train_rounds = np.sort(train_all["round_index"].unique())
        calibration_cut = train_rounds[-validation_rounds]
        fit_part = train_all[train_all["round_index"] < calibration_cut]
        calib_part = train_all[train_all["round_index"] >= calibration_cut]

        if len(fit_part) < 40 or len(calib_part) < 20:
            raise ValueError(
                f"Split leaves {len(fit_part)} fitting rows and {len(calib_part)} calibration "
                "rows. Both are too small to fit a reach model responsibly."
            )

        notes: list[str] = []

        x_fit = self._design(fit_part, fitting=True)
        y_fit = fit_part["reach"].to_numpy()
        constraints = self._constraints(x_fit.shape[1])

        provisional = lgb.LGBMRegressor(monotone_constraints=constraints, **self.params)
        provisional.fit(x_fit, y_fit)

        # The interval width is the shape of the standardised residual, and the
        # shape has to be read off rows that taught neither model. Fitting the
        # scale model and then dividing that model's own training residuals by its
        # own predictions is the mistake it looks like: the scale model has partly
        # learned those residuals, so the standardised values come out too small
        # and every interval built from them runs narrow. The failure does not
        # show up where it is made - coverage on the calibration slice looks
        # correct - it shows up on the rounds a programme is actually planning.
        #
        # So the calibration slice is split again, by round. The earlier rounds
        # teach the spread, the later rounds say what that spread is worth out of
        # sample, and neither has been seen by the centre model.
        calibration_rounds = np.sort(calib_part["round_index"].unique())
        scale_part, shape_part = calib_part, calib_part
        leaked = True
        if len(calibration_rounds) >= 2:
            shape_cut = calibration_rounds[-1]
            candidate_scale = calib_part[calib_part["round_index"] < shape_cut]
            candidate_shape = calib_part[calib_part["round_index"] >= shape_cut]
            if len(candidate_scale) >= 20 and len(candidate_shape) >= 20:
                scale_part, shape_part, leaked = candidate_scale, candidate_shape, False

        x_scale = self._design(scale_part)
        self.scale = lgb.LGBMRegressor(**{**self.params, "n_estimators": 200})
        self.scale.fit(x_scale, np.abs(scale_part["reach"].to_numpy() - provisional.predict(x_scale)))

        x_shape = self._design(shape_part)
        shape_residual = shape_part["reach"].to_numpy() - provisional.predict(x_shape)
        shape_scale = np.maximum(self.scale.predict(x_shape), MIN_SCALE)
        self.residual_quantiles = np.quantile(shape_residual / shape_scale, self.quantiles)

        if leaked:
            notes.append(
                "The calibration slice held only one round, so the spread model and the "
                "interval shape were read off the same rows. The intervals will run "
                "narrower than they claim. Widen the panel before quoting their coverage."
            )

        # Refit the centre on everything before the holdout.
        x_train = self._design(train_all)
        self.centre = lgb.LGBMRegressor(monotone_constraints=constraints, **self.params)
        self.centre.fit(x_train, train_all["reach"].to_numpy())

        # -- scoring -------------------------------------------------------
        x_valid = self._design(valid)
        y_valid = valid["reach"].to_numpy()
        predictions = self._predict_matrix(x_valid)
        median = predictions[:, self.quantiles.index(0.50)]

        coverage: dict[str, float] = {}
        for low, high in ((0.05, 0.95), (0.10, 0.90), (0.25, 0.75)):
            if low in self.quantiles and high in self.quantiles:
                lo = predictions[:, self.quantiles.index(low)]
                hi = predictions[:, self.quantiles.index(high)]
                coverage[f"{(high - low) * 100:.0f}%"] = float(
                    np.mean((y_valid >= lo) & (y_valid <= hi))
                )

        baseline = valid["reach_prev"].to_numpy()
        has_baseline = np.isfinite(baseline)
        if has_baseline.sum() >= 20:
            baseline_mae = float(np.mean(np.abs(y_valid[has_baseline] - baseline[has_baseline])))
            model_mae_matched = float(np.mean(np.abs(y_valid[has_baseline] - median[has_baseline])))
            skill = 1.0 - model_mae_matched / baseline_mae if baseline_mae > 0 else 0.0
        else:
            skill = float("nan")
            notes.append(
                "Too few validation rows had a previous round, so skill against the "
                "persistence baseline could not be computed."
            )

        importance = dict(
            zip(x_train.columns, self.centre.booster_.feature_importance(importance_type="gain"))
        )
        total = sum(importance.values()) or 1.0
        importance = {
            k: float(v / total)
            for k, v in sorted(importance.items(), key=lambda kv: -kv[1])
        }

        if valid["unit_id"].nunique() < 30:
            notes.append(
                f"Validation covers only {valid['unit_id'].nunique()} units, so interval "
                "coverage is itself imprecise. Treat the calibration verdict as provisional."
            )
        if float(np.std(y_valid)) < 0.02:
            notes.append(
                "Held-out reach barely varies, so the error looks small for reasons "
                "unrelated to model skill."
            )
        if not np.isnan(skill) and skill <= 0.0:
            notes.append(
                "The model does not beat assuming next round repeats last round. Prefer the "
                "persistence baseline and treat the learned layer as unproven here."
            )

        self.report = ReachModelReport(
            n_train=len(train_all),
            n_calibration=len(calib_part),
            n_validation=len(valid),
            mae=float(np.mean(np.abs(y_valid - median))),
            rmse=float(np.sqrt(np.mean((y_valid - median) ** 2))),
            pinball_loss=float(
                np.mean([pinball_loss(y_valid, predictions[:, i], q) for i, q in enumerate(self.quantiles)])
            ),
            skill_vs_persistence=float(skill),
            interval_coverage=coverage,
            feature_importance=importance,
            notes=notes,
        )
        return self.report

    # -- prediction --------------------------------------------------------

    def _predict_matrix(self, design: pd.DataFrame) -> np.ndarray:
        """Rebuild the quantiles from centre, scale and residual shape."""
        if self.centre is None or self.scale is None or self.residual_quantiles is None:
            raise RuntimeError("Model is not fitted. Call fit first.")
        centre = self.centre.predict(design)
        scale = np.maximum(self.scale.predict(design), MIN_SCALE)
        matrix = centre[:, None] + scale[:, None] * self.residual_quantiles[None, :]
        # Reach is a share, and the quantiles must not cross after clipping.
        return np.sort(np.clip(matrix, 0.0, 1.0), axis=1)

    def predict_quantiles(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predicted reach quantiles for each row.

        Args:
            features: Rows built by ``features.build_features``.

        Returns:
            A frame indexed like ``features``, one column per quantile, named
            ``q05``, ``q50`` and so on.
        """
        matrix = self._predict_matrix(self._design(features))
        names = [f"q{int(round(q * 100)):02d}" for q in self.quantiles]
        return pd.DataFrame(matrix, columns=names, index=features.index)

    def sample_reach(
        self, features: pd.DataFrame, n_draws: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw from the predictive reach distribution of each row.

        The quantiles define an empirical inverse cumulative distribution. Drawing
        a uniform and interpolating between neighbouring quantiles samples from it
        without imposing a parametric shape, which matters because reach is
        bounded and skewed.

        Args:
            features: Rows to sample for.
            n_draws: Draws per row.
            rng: Random generator, passed in so runs are reproducible.

        Returns:
            An array of shape ``(len(features), n_draws)`` with values in ``[0, 1]``.
        """
        matrix = self.predict_quantiles(features).to_numpy()
        levels = np.asarray(self.quantiles)
        uniforms = rng.uniform(levels[0], levels[-1], size=(matrix.shape[0], n_draws))
        draws = np.empty_like(uniforms)
        for i in range(matrix.shape[0]):
            draws[i] = np.interp(uniforms[i], levels, matrix[i])
        return np.clip(draws, 0.0, 1.0)
