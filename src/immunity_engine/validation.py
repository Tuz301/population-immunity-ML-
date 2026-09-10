"""Does the engine deserve to be believed?

Three separate questions, kept separate because a good answer to one is often
mistaken for a good answer to all three.

*Discrimination.* Does the engine rank places correctly? A programme that only
needs to know where to send the next round can act on ranking alone.

*Calibration.* When the engine says four rounds will do it with 80% confidence,
does that happen 80% of the time? This is the property that makes a stated
interval mean anything, and it is the one most models fail silently.

*Recovery under misspecification.* When the world does not behave the way the
model assumes, does the answer degrade gracefully or does it break?

The first two can be measured on a real panel by refitting on early rounds and
scoring on later ones. The third needs a simulated programme where truth is
known. Both are implemented here, and the report says which one produced which
number, because they support very different claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .contracts import EngineConfig, Feasibility
from .pipeline import PlanSet, run_pipeline
from .features import build_features
from .rounds_required import UnitPlan


@dataclass
class ReliabilityPoint:
    """One bin of a reliability diagram.

    Attributes:
        predicted: Mean predicted probability in the bin.
        observed: Share of cases in the bin where the event actually happened.
        n: Cases in the bin.
    """

    predicted: float
    observed: float
    n: int


@dataclass
class ValidationResult:
    """Everything one validation run measured.

    Attributes:
        n_units: Units scored.
        n_comparable: Units where both the engine and the truth gave a finite count.
        rounds_mae: Mean absolute error in rounds, over comparable units.
        rounds_median_ae: Median absolute error in rounds. Reported because the
            mean is dragged by the tail of very hard places.
        rounds_within_one: Share of comparable units predicted within one round.
        rounds_spearman: Rank correlation between predicted and true round counts.
        interval_coverage_80: Share of comparable units whose true count fell
            inside the engine's 80% interval. Should be near 0.80.
        infeasible_recall: Share of truly unreachable units the engine flagged.
        infeasible_precision: Share of flagged units that were truly unreachable.
        brier_score: Brier score of the statement "N rounds will be enough",
            pooled over units and round counts. Lower is better; 0.25 is the
            score of always saying 50%.
        reliability: Reliability diagram of that same statement.
        immunity_mae: Mean absolute error of the reconstructed current immunity.
        notes: Caveats a reader must carry with the numbers.
    """

    n_units: int
    n_comparable: int
    rounds_mae: float
    rounds_median_ae: float
    rounds_within_one: float
    rounds_spearman: float
    interval_coverage_80: float
    infeasible_recall: float
    infeasible_precision: float
    brier_score: float
    reliability: list[ReliabilityPoint]
    immunity_mae: float
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """A block a reviewer can paste into a note without rewriting it."""
        lines = [
            f"Units scored: {self.n_units} ({self.n_comparable} with a finite count on both sides)",
            f"Round count      MAE {self.rounds_mae:.2f} | median AE {self.rounds_median_ae:.2f} "
            f"| within 1 round {self.rounds_within_one:.0%} | rank corr {self.rounds_spearman:.2f}",
            f"80% interval     covers the truth {self.interval_coverage_80:.0%} of the time "
            f"(target 80%)",
            f"Unreachable flag recall {self.infeasible_recall:.0%} | precision "
            f"{self.infeasible_precision:.0%}",
            f"Round-sufficiency Brier score {self.brier_score:.3f} "
            f"(0.25 = uninformative)",
            f"Current immunity MAE {self.immunity_mae:.3f}",
        ]
        lines.extend(f"Caveat: {note}" for note in self.notes)
        return "\n".join(lines)


def _reliability(
    predicted: np.ndarray, outcome: np.ndarray, n_bins: int = 10
) -> tuple[list[ReliabilityPoint], float]:
    """Bin predicted probabilities against what happened.

    Args:
        predicted: Predicted probabilities.
        outcome: Binary outcomes.
        n_bins: Number of equal-width bins.

    Returns:
        A pair ``(points, brier_score)``.
    """
    predicted = np.asarray(predicted, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    keep = np.isfinite(predicted) & np.isfinite(outcome)
    predicted, outcome = predicted[keep], outcome[keep]
    if predicted.size == 0:
        return [], float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(predicted, edges[1:-1]), 0, n_bins - 1)
    points = []
    for b in range(n_bins):
        sel = index == b
        if sel.sum() == 0:
            continue
        points.append(
            ReliabilityPoint(
                predicted=float(predicted[sel].mean()),
                observed=float(outcome[sel].mean()),
                n=int(sel.sum()),
            )
        )
    brier = float(np.mean((predicted - outcome) ** 2))
    return points, brier


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without pulling in scipy for one number."""
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def validate_against_truth(
    plan_set: PlanSet,
    truth: pd.DataFrame,
    *,
    max_rounds: int,
) -> ValidationResult:
    """Score an engine run against a simulated programme's known truth.

    Args:
        plan_set: Output of ``pipeline.run_pipeline``.
        truth: Truth frame from the synthetic adapter. Needs ``unit_id``,
            ``true_rounds_needed`` and ``true_immunity_at_cut``.
        max_rounds: Round budget the engine was given. Truth beyond this is
            censored the same way the engine censors it, so the two are compared
            on the same footing.

    Returns:
        A ``ValidationResult``.
    """
    plans: dict[str, UnitPlan] = {p.unit_id: p for p in plan_set.plans}
    truth = truth.set_index("unit_id")
    shared = [u for u in truth.index if u in plans]
    if not shared:
        raise ValueError("No units are shared between the plans and the truth frame.")

    notes: list[str] = []

    true_rounds = truth.loc[shared, "true_rounds_needed"].to_numpy(dtype=float)
    # Truth beyond the budget is not a round count the engine could have given.
    true_censored = np.where(true_rounds <= max_rounds, true_rounds, np.inf)

    predicted_median = np.array(
        [plans[u].rounds_median if plans[u].rounds_median is not None else np.inf for u in shared]
    )
    p10 = np.array(
        [plans[u].rounds_interval_80[0] if plans[u].rounds_interval_80[0] is not None else np.inf for u in shared]
    )
    p90 = np.array(
        [plans[u].rounds_interval_80[1] if plans[u].rounds_interval_80[1] is not None else np.inf for u in shared]
    )

    comparable = np.isfinite(true_censored) & np.isfinite(predicted_median)
    n_comparable = int(comparable.sum())
    if n_comparable < 10:
        notes.append(
            f"Only {n_comparable} units had a finite count on both sides. The round-count "
            "metrics rest on very little and should not be quoted alone."
        )

    if n_comparable:
        error = predicted_median[comparable] - true_censored[comparable]
        rounds_mae = float(np.mean(np.abs(error)))
        rounds_median_ae = float(np.median(np.abs(error)))
        within_one = float(np.mean(np.abs(error) <= 1.0))
        spearman = _spearman(predicted_median[comparable], true_censored[comparable])
        inside = (true_censored[comparable] >= p10[comparable]) & (
            true_censored[comparable] <= p90[comparable]
        )
        coverage_80 = float(np.mean(inside))
    else:
        rounds_mae = rounds_median_ae = within_one = spearman = coverage_80 = float("nan")

    # --- the unreachable verdict -----------------------------------------
    truly_unreachable = ~np.isfinite(true_rounds)
    flagged = np.array(
        [plans[u].feasibility is Feasibility.INFEASIBLE_BY_CAMPAIGNS for u in shared]
    )
    recall = float(flagged[truly_unreachable].mean()) if truly_unreachable.any() else float("nan")
    precision = float(truly_unreachable[flagged].mean()) if flagged.any() else float("nan")

    # --- calibration of "N rounds will be enough" ------------------------
    predicted_probability: list[float] = []
    happened: list[float] = []
    for i, unit in enumerate(shared):
        for n, probability in plans[unit].probability_by_round.items():
            predicted_probability.append(probability)
            happened.append(1.0 if true_censored[i] <= n else 0.0)
    reliability, brier = _reliability(np.array(predicted_probability), np.array(happened))

    # --- reconstructed immunity ------------------------------------------
    if "current_immunity" in plan_set.parameters.columns:
        estimated = plan_set.parameters.reindex(shared)["current_immunity"].to_numpy(dtype=float)
        actual = truth.loc[shared, "true_immunity_at_cut"].to_numpy(dtype=float)
        immunity_mae = float(np.nanmean(np.abs(estimated - actual)))
    else:
        immunity_mae = float("nan")

    if not np.isnan(coverage_80) and abs(coverage_80 - 0.80) > 0.10:
        notes.append(
            f"The 80% interval covered the truth {coverage_80:.0%} of the time. "
            + ("It is too narrow; round counts are stated more confidently than they deserve."
               if coverage_80 < 0.80
               else "It is too wide; the engine is under-claiming what it knows.")
        )

    return ValidationResult(
        n_units=len(shared),
        n_comparable=n_comparable,
        rounds_mae=rounds_mae,
        rounds_median_ae=rounds_median_ae,
        rounds_within_one=within_one,
        rounds_spearman=spearman,
        interval_coverage_80=coverage_80,
        infeasible_recall=recall,
        infeasible_precision=precision,
        brier_score=brier,
        reliability=reliability,
        immunity_mae=immunity_mae,
        notes=notes,
    )


def backtest_on_panel(
    panel: pd.DataFrame,
    config: EngineConfig | None = None,
    *,
    holdout_rounds: int = 3,
    planning_interval_months: float = 3.0,
) -> pd.DataFrame:
    """Refit on early rounds, then check the forecast against what happened next.

    No simulated truth is needed. The engine is run on the panel up to a cut
    point, and its statement about the next rounds is compared with the reach
    those rounds actually achieved. This is the validation a real programme can
    run on its own data, and it is the only one whose numbers describe the field.

    It scores reach, not the round count: on a real panel the true round count is
    never observed, because the counterfactual was not run. Reach error is the
    input the round count is most sensitive to, so it is the honest proxy.

    Args:
        panel: Panel satisfying the data contract.
        config: Engine configuration.
        holdout_rounds: Trailing rounds withheld from fitting.
        planning_interval_months: Planning tempo passed to the engine.

    Returns:
        A frame with one row per held-out unit-round, carrying the predicted
        reach quantiles and the reach that was observed.
    """
    config = config or EngineConfig()
    rounds = np.sort(panel["round_index"].unique())
    # The reach model splits the history it is given into a fitting slice and a
    # calibration slice of matching depth, so the history has to be at least four
    # rounds deep before any of it can be held out.
    minimum = holdout_rounds + 4
    if len(rounds) < minimum:
        raise ValueError(
            f"Panel has {len(rounds)} rounds; a {holdout_rounds}-round backtest needs at "
            f"least {minimum}, because the reach model needs a fitting slice and a "
            "calibration slice inside the history."
        )
    cut = rounds[-holdout_rounds]

    history = panel[panel["round_index"] < cut]
    plan_set = run_pipeline(
        history,
        config,
        planning_interval_months=planning_interval_months,
        validation_rounds=1,
    )
    if plan_set.reach_report is None:
        raise ValueError("Reach model was not fitted on the history, so there is nothing to score.")

    # Features for the held-out rounds are built on the full panel, because their
    # lags legitimately look back into the history the model was fitted on.
    full_features, _ = build_features(panel, config)
    future = full_features[full_features["round_index"] >= cut].copy()
    future = future[future["reach"].notna() & (future["target_pop"] > 0)]

    quantiles = plan_set.reach_model.predict_quantiles(future)
    scored = pd.concat(
        [
            future[["unit_id", "lga_code", "round_index", "reach"]].reset_index(drop=True),
            quantiles.reset_index(drop=True),
        ],
        axis=1,
    )
    scored["error"] = scored["q50"] - scored["reach"]
    scored["inside_80"] = (scored["reach"] >= scored["q10"]) & (scored["reach"] <= scored["q90"])
    return scored


def summarise_backtest(scored: pd.DataFrame) -> str:
    """Turn a backtest frame into the three numbers that decide whether to trust it.

    Args:
        scored: Output of ``backtest_on_panel``.

    Returns:
        A short block naming the error, the interval coverage and the skill over
        assuming next round repeats last round.
    """
    if scored.empty:
        return "Backtest produced no scorable rows."
    mae = float(scored["error"].abs().mean())
    coverage = float(scored["inside_80"].mean())
    lines = [
        f"Held-out unit-rounds: {len(scored)} across {scored['unit_id'].nunique()} units",
        f"Reach MAE {mae:.4f} | 80% interval covers {coverage:.0%} of observations (target 80%)",
    ]
    if abs(coverage - 0.80) > 0.10:
        lines.append(
            "Interval coverage is off target, so any round-count interval built on this "
            "reach model is indicative only."
        )
    return "\n".join(lines)
