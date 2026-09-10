"""Wiring: from a reported panel to a campaign requirement per unit.

The order matters and is not arbitrary.

1. Validate the panel and record every fallback that had to be used.
2. Build leak-free features.
3. Fit the reach model on history and score it on the rounds it did not see.
4. Estimate the unreachable core, the spread of reach and its stickiness.
5. Reconstruct current immunity by replaying the observed rounds through the
   stock model, starting from the immunity that routine immunisation alone
   sustains.
6. Invert: how many further rounds clear the threshold, and can any number.

Step 5 is where most of the honesty lives. Current immunity is never measured
directly at settlement grain, so it is reconstructed rather than read. The
reconstruction starts from the routine-immunisation equilibrium because that is
the level a place returns to when campaigns stop, which makes the baseline a
property of the health system rather than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .contracts import (
    AccessibilityStatus,
    EngineConfig,
    Feasibility,
    ProvenanceLog,
    validate_panel,
)
from .features import build_features
from .heterogeneity import (
    estimate_concentration,
    estimate_persistence,
    estimate_persistence_from_reach,
    estimate_reach_wobble,
    unreachable_core,
)
from .reach_model import ReachModel, ReachModelReport
from .rounds_required import UnitInputs, UnitPlan, solve_unit
from .vectorised import build_reach_grid


@dataclass
class PlanSet:
    """Everything one run of the engine produced.

    Attributes:
        plans: One plan per unit.
        reach_report: Out-of-sample performance of the reach model.
        reach_model: The fitted reach model, or None when it could not be fitted.
            Kept so a backtest can score it on later rounds without refitting.
        provenance: Every fallback and assumption used.
        parameters: Per-unit estimated parameters, kept so a reviewer can audit
            the inputs rather than only the answer.
        notes: Programme-level notes, including how stickiness was estimated.
    """

    plans: list[UnitPlan]
    reach_report: ReachModelReport | None
    reach_model: ReachModel | None
    provenance: ProvenanceLog
    parameters: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Flatten the plans into one row per unit."""
        if not self.plans:
            return pd.DataFrame()
        rows = []
        for plan in self.plans:
            row = plan.to_dict()
            row.pop("probability_by_round")
            low, high = row.pop("rounds_interval_80")
            row["rounds_p10"] = low
            row["rounds_p90"] = high
            rows.append(row)
        return pd.DataFrame(rows)

    def escalations(self) -> pd.DataFrame:
        """Units where more rounds is the wrong instrument.

        This is the table that changes what a programme does, so it is a first
        class output rather than a filter someone has to remember to apply.
        """
        frame = self.to_frame()
        if frame.empty:
            return frame
        return (
            frame[frame["feasibility"] == Feasibility.INFEASIBLE_BY_CAMPAIGNS.value]
            .sort_values("immunity_ceiling_median")
            .reset_index(drop=True)
        )


def estimate_unit_parameters(
    features: pd.DataFrame,
    panel: pd.DataFrame,
    config: EngineConfig,
    log: ProvenanceLog,
) -> tuple[pd.DataFrame, list[str]]:
    """Estimate the unreachable core, reach spread and stickiness for each unit.

    Args:
        features: Feature frame from ``features.build_features``.
        panel: The original panel, for columns the features drop.
        config: Engine configuration.
        log: Provenance log to append to.

    Returns:
        A pair ``(parameters, notes)``. ``parameters`` is indexed by ``unit_id``.
    """
    notes: list[str] = []
    latest = features.sort_values("round_index").groupby("unit_id").tail(1).set_index("unit_id")


    # --- unreachable core -------------------------------------------------
    # A settlement closed in some rounds and open in others is reachable; it was
    # simply not reached every time, and the reach model already accounts for
    # that. Only settlements closed in every round, or all but one, contribute
    # children that no number of rounds can touch. Treating intermittent closure
    # as permanent was the single largest source of false "campaigns cannot do
    # this" verdicts in testing, and it sends a programme to negotiate access it
    # already has.
    inaccessible = pd.Series(0.0, index=latest.index)
    closed_flag: pd.Series | None = None
    if "is_inaccessible" in panel.columns:
        closed_flag = panel["is_inaccessible"].astype(bool)
    elif "accessibility_status" in panel.columns:
        closed_flag = panel["accessibility_status"].eq(AccessibilityStatus.INACCESSIBLE.value)

    if closed_flag is not None:
        observed = panel.assign(closed=closed_flag).groupby("unit_id")["closed"].agg(
            ["mean", "size"]
        )
        persistent = observed["mean"] >= (observed["size"] - 1).clip(lower=1) / observed["size"]
        inaccessible = (
            observed["mean"].where(persistent, 0.0).reindex(latest.index).fillna(0.0).astype(float)
        )
        intermittent = int((~persistent & (observed["mean"] > 0)).sum())
        if intermittent:
            notes.append(
                f"{intermittent} units were closed in some rounds but not all. Their closures "
                "count against reach, not against the permanently unreachable core."
            )
    else:
        log.record(
            "is_inaccessible",
            "No accessibility data. The unreachable core comes only from refusals and mobility.",
            "Feasibility verdicts are optimistic wherever access is the real constraint.",
        )

    if "refusal_count" in panel.columns and "target_pop" in panel.columns:
        refusal = (
            panel.assign(rate=panel["refusal_count"] / panel["target_pop"].replace(0, np.nan))
            .groupby("unit_id")["rate"]
            .mean()
            .reindex(latest.index)
            .fillna(config.refusal_hard_core)
        )
        # Only refusals that repeat every round belong in the unreachable core.
        # Treat a third of the average refusal rate as hard core; the rest is
        # persuadable between rounds.
        refusal = (refusal * 0.33).clip(0.0, 0.25)
    else:
        refusal = pd.Series(config.refusal_hard_core, index=latest.index)

    nomadic = (
        panel.groupby("unit_id")["nomadic_share"].mean().reindex(latest.index).fillna(0.0)
        if "nomadic_share" in panel.columns
        else pd.Series(0.0, index=latest.index)
    )

    pi_zero = pd.Series(
        [
            unreachable_core(
                inaccessible_share=float(inaccessible.iloc[i]),
                refusal_share=float(refusal.iloc[i]),
                nomadic_share=float(nomadic.iloc[i]),
            )
            for i in range(len(latest))
        ],
        index=latest.index,
    )

    # --- spread of reach across children ----------------------------------
    # Estimated within each LGA from the spread across its settlements. Reach
    # varies between children for the same reasons it varies between
    # settlements, so the between-settlement spread is the observable proxy.
    lot_size = None
    if "verified_sample_n" in panel.columns:
        median_lot = panel["verified_sample_n"].median()
        lot_size = int(median_lot) if np.isfinite(median_lot) else None
    if lot_size is None:
        lot_size = config.default_lqas_lot_size
        log.record(
            "verified_sample_n",
            f"Absent. Verification lots assumed to be {lot_size}.",
            "The correction for measurement noise in reach spread rests on that assumption.",
        )

    # Reach is divided by the reachable share, because the Beta component
    # describes reach among the children a campaign can get to, not among all of
    # them; the rest are already held in pi_zero.
    #
    # Rounds where a settlement was closed are deliberately kept. A settlement
    # shut in some rounds and open in others really does give its children lower
    # reach than a settlement always open, and that is inequality between
    # children, which is exactly what this parameter measures. Only *permanent*
    # closure was moved into pi_zero, so nothing is double counted by keeping
    # these rounds; dropping them was measured to make the engine roughly ten
    # points over-confident about how few rounds would do.
    reach_for_kappa = features[["unit_id", "lga_code", "round_index", "reach"]].copy()
    reach_for_kappa["reachable_share"] = (
        1.0 - reach_for_kappa["unit_id"].map(pi_zero).fillna(0.0)
    )
    reach_for_kappa["adjusted"] = (
        reach_for_kappa["reach"] / reach_for_kappa["reachable_share"].clip(lower=1e-3)
    ).clip(0.0, 1.0)

    kappa_by_lga: dict[str, float] = {}
    for lga, group in reach_for_kappa.groupby("lga_code"):
        recent = group.sort_values("round_index").groupby("unit_id").tail(2)
        kappa_value, _ = estimate_concentration(
            recent["adjusted"].to_numpy(), verification_lot_size=lot_size
        )
        kappa_by_lga[lga] = kappa_value
    kappa = latest["lga_code"].map(kappa_by_lga).fillna(8.0)
    notes.append(
        f"Reach concentration estimated per LGA from between-settlement spread, with "
        f"verification sampling noise removed; median kappa "
        f"{np.median(list(kappa_by_lga.values())):.1f} across {len(kappa_by_lga)} LGAs."
    )

    # --- round-to-round movement in reach ---------------------------------
    # Drift is where reach is heading. This is how far it strays on the way, and
    # the two are separate: a programme can hold a steady average and still miss
    # one round badly enough to change how many rounds it needs. Both reported
    # streams are used, because the movement they agree on is the round and the
    # movement only one of them shows is reporting noise.
    # Both streams are needed: what the two agree on is the round, what only one
    # of them shows is reporting noise. Without verification there is no second
    # opinion, and the single reported series cannot tell them apart.
    if "verified_vaccinated" in panel.columns and "admin_vaccinated" in panel.columns:
        denominator = panel["target_pop"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            admin_reach = np.where(
                denominator > 0, panel["admin_vaccinated"] / denominator, np.nan
            )
            verified_reach = np.where(
                denominator > 0, panel["verified_vaccinated"] / denominator, np.nan
            )
        wobble, wobble_note = estimate_reach_wobble(
            panel["unit_id"].to_numpy(),
            panel["round_index"].to_numpy(),
            admin_reach,
            verified_reach,
            cap=config.max_reach_wobble_cv,
        )
    else:
        wobble, wobble_note = 0.0, (
            "Round-to-round reach movement not estimated: the panel carries only one "
            "reported stream, so campaign movement cannot be told from reporting noise. "
            "Reach is held steady between rounds and the round count has a thinner "
            "middle than the field does."
        )
    notes.append(wobble_note)

    # --- stickiness -------------------------------------------------------
    usable = features[features["reach"].notna()]
    rho, rho_note = estimate_persistence_from_reach(
        usable["unit_id"].to_numpy(),
        usable["round_index"].to_numpy(),
        usable["reach"].to_numpy(),
        verification_lot_size=lot_size,
        campaign_wobble_cv=wobble,
    )
    notes.append(rho_note)

    # Cross-check against the pass-or-fail estimator the portal already publishes.
    # The two disagree when reach is spread out inside the failing group, and the
    # gap is worth seeing rather than hiding.
    below = features.assign(miss=(features["reach"] < 0.80).astype(float))
    per_unit = below.groupby("unit_id").agg(misses=("miss", "sum"), targeted=("miss", "size"))
    rho_threshold, _ = estimate_persistence(
        per_unit["misses"].to_numpy(), per_unit["targeted"].to_numpy()
    )
    if abs(rho_threshold - rho) > 0.15:
        notes.append(
            f"The chronic-miss pass-or-fail estimator gives rho = {rho_threshold:.2f} against "
            f"{rho:.2f} from the continuous series. The continuous estimate is used; "
            "thresholding discards how far below the line a settlement sits."
        )

    # --- denominator inflation -------------------------------------------
    # Administrative coverage above 100% is arithmetically impossible unless the
    # denominator is too large, so the excess is a lower bound on the inflation.
    admin_rate = panel["admin_vaccinated"] / panel["target_pop"].replace(0, np.nan)
    peak = admin_rate.groupby(panel["unit_id"]).max().reindex(latest.index)
    inflation = peak.clip(lower=1.0).fillna(1.0)
    inflation = inflation.where(inflation > 1.0, 1.0)
    n_inflated = int((inflation > 1.0).sum())
    if n_inflated:
        notes.append(
            f"{n_inflated} units reported administrative coverage above 100% in at least one "
            "round. Their denominators are treated as inflated by at least that excess."
        )

    # --- reach drift across rounds ---------------------------------------
    drift = 0.0
    if config.estimate_reach_drift:
        drift, drift_note = _estimate_reach_drift(features, config)
        notes.append(drift_note)

    parameters = pd.DataFrame(
        {
            "reach_drift_per_round": drift,
            "reach_wobble_cv": wobble,
            "pi_zero": pi_zero,
            "kappa": kappa,
            "rho": rho,
            "denominator_inflation": inflation,
            "lga_code": latest["lga_code"],
            "state_code": latest["state_code"],
            "grain": latest["grain"],
            "target_pop": latest["target_pop"],
            "denominator_basis": latest["denominator_basis"],
        }
    )
    return parameters, notes


def _estimate_reach_drift(features: pd.DataFrame, config: EngineConfig) -> tuple[float, str]:
    """Measure how reach changes from one round to the next, net of place.

    Teams and communities tire. A model that assumes reach is stationary will
    forecast too few rounds wherever that is true, which is most places. The slope
    is measured within unit, so it is not confounded by which settlements happened
    to be targeted in which round.

    Args:
        features: Feature frame carrying reach and round index.
        config: Engine configuration, for the cap on plausible drift.

    Returns:
        A pair ``(drift_per_round, note)``.
    """
    usable = features[features["reach"].notna()][["unit_id", "round_index", "reach"]]
    if usable["unit_id"].nunique() < 20 or usable["round_index"].nunique() < 3:
        return 0.0, "Reach drift not estimated: too few units or rounds. Reach assumed stationary."

    # Within-unit demeaning removes place, leaving the common trend across rounds.
    unit_mean = usable.groupby("unit_id")["reach"].transform("mean")
    round_mean = usable.groupby("unit_id")["round_index"].transform("mean")
    x = (usable["round_index"] - round_mean).to_numpy(dtype=float)
    y = (usable["reach"] - unit_mean).to_numpy(dtype=float)
    denominator = float(np.sum(x * x))
    if denominator <= 0:
        return 0.0, "Reach drift not estimated: no within-unit variation in round index."

    slope = float(np.sum(x * y) / denominator)
    level = float(usable["reach"].mean())
    if level <= 0:
        return 0.0, "Reach drift not estimated: mean reach is zero."

    proportional = float(
        np.clip(slope / level, -config.max_reach_drift_per_round, config.max_reach_drift_per_round)
    )
    direction = "falls" if proportional < 0 else "rises"
    return proportional, (
        f"Reach {direction} by {abs(proportional):.1%} per round within unit "
        f"({slope:+.4f} reach points on a mean of {level:.3f}). Projected forward, with a "
        f"floor at {config.reach_drift_floor:.0%} of current reach."
    )


def estimate_current_immunity(
    features: pd.DataFrame,
    parameters: pd.DataFrame,
    panel: pd.DataFrame,
    config: EngineConfig,
) -> pd.DataFrame:
    """Reconstruct immunity now by replaying the observed rounds.

    Immunity at settlement grain is not measured. It is rebuilt: start each unit
    at the immunity routine immunisation alone sustains, then apply each observed
    round with the reach that round actually achieved, letting births refill the
    stock in between.

    The starting point is not a free parameter. With no campaigns the susceptible
    stock settles where births and ageing balance, which puts population immunity
    at the share of newborns routine immunisation protects. That is the floor a
    place returns to, so it is the right place to start a replay from.

    Args:
        features: Feature frame carrying the per-round reach.
        parameters: Per-unit parameters from ``estimate_unit_parameters``.
        panel: The original panel.
        config: Engine configuration.

    Returns:
        A frame indexed by ``unit_id`` with ``current_immunity``,
        ``current_immunity_sd``, ``births_per_month`` and ``ri_protection``.
    """
    ri = (
        panel.groupby("unit_id")["ri_coverage"].mean()
        if "ri_coverage" in panel.columns
        else pd.Series(config.default_ri_coverage, index=parameters.index)
    )
    ri = ri.reindex(parameters.index).fillna(config.default_ri_coverage).clip(0.0, 0.98)

    if "births_per_month" in panel.columns:
        births = panel.groupby("unit_id")["births_per_month"].mean().reindex(parameters.index)
    else:
        births = pd.Series(np.nan, index=parameters.index)
    births = births.fillna(
        parameters["target_pop"] / parameters["denominator_inflation"] / config.target_age_months
    ).clip(lower=0.0)

    # Round spacing from the dates, so a replay uses the tempo that was run.
    dates = (
        features.sort_values("round_index")
        .groupby("unit_id")["round_start"]
        .apply(lambda s: pd.to_datetime(s).diff().dt.days.mean() / 30.44)
    )
    spacing = dates.reindex(parameters.index).fillna(6.0).clip(0.5, 36.0)

    reach_history = (
        features.sort_values("round_index").groupby("unit_id")["reach"].apply(list)
    ).reindex(parameters.index)

    immunity = np.empty(len(parameters))
    immunity_sd = np.empty(len(parameters))

    for i, unit in enumerate(parameters.index):
        history = [r for r in (reach_history.iloc[i] or []) if np.isfinite(r)]
        pi0 = float(parameters["pi_zero"].iloc[i])
        cohort = max(
            float(parameters["target_pop"].iloc[i]) / float(parameters["denominator_inflation"].iloc[i]),
            1.0,
        )
        baseline = float(ri.iloc[i])

        if not history:
            immunity[i], immunity_sd[i] = baseline, 0.12
            continue

        # Three replays: the reach as measured, and plus or minus the sampling
        # error of a small verification lot. The spread becomes the uncertainty.
        results = []
        for shift in (-1.0, 0.0, 1.0):
            noise = shift * np.sqrt(0.25 / config.default_lqas_lot_size)
            results.append(
                _replay(
                    history=np.clip(np.array(history) + noise, 0.0, 1.0),
                    pi_zero=pi0,
                    kappa=float(parameters["kappa"].iloc[i]),
                    rho=float(parameters["rho"].iloc[i]),
                    take=config.per_dose_take_mean,
                    baseline_immunity=baseline,
                    cohort=cohort,
                    births=float(births.iloc[i]) / float(parameters["denominator_inflation"].iloc[i]),
                    ri_protection=baseline,
                    interval_months=float(spacing.iloc[i]),
                    age_band_months=config.target_age_months,
                )
            )
        immunity[i] = results[1]
        immunity_sd[i] = max((results[2] - results[0]) / 2.0, 0.02)

    return pd.DataFrame(
        {
            "current_immunity": np.clip(immunity, 0.0, 0.999),
            "current_immunity_sd": np.clip(immunity_sd, 0.01, 0.25),
            "births_per_month": births.to_numpy(),
            "ri_protection": ri.to_numpy(),
            "interval_observed_months": spacing.to_numpy(),
        },
        index=parameters.index,
    )


def _replay(
    *,
    history: np.ndarray,
    pi_zero: float,
    kappa: float,
    rho: float,
    take: float,
    baseline_immunity: float,
    cohort: float,
    births: float,
    ri_protection: float,
    interval_months: float,
    age_band_months: int,
) -> float:
    """Push one unit through its observed rounds and return immunity at the end."""
    mu = np.clip(history / max(1.0 - pi_zero, 1e-6), 1e-4, 1.0 - 1e-4)
    nodes, weights = build_reach_grid(
        mu, np.full_like(mu, kappa), np.full_like(mu, pi_zero)
    )
    # Every round shares one population distribution for births; the per-round
    # grids differ only in how well that round reached those children.
    population_weights = weights[0]

    susceptible = cohort * (1.0 - baseline_immunity) * population_weights
    total = cohort
    decay = float(np.exp(-interval_months / age_band_months))
    inflow = age_band_months * (1.0 - decay)

    for r in range(len(history)):
        total = total * decay + births * inflow
        susceptible = susceptible * decay + births * (1.0 - ri_protection) * inflow * population_weights
        susceptible = susceptible * (1.0 - take * nodes[r])
        remaining = susceptible.sum()
        susceptible = rho * susceptible + (1.0 - rho) * remaining * population_weights

    return float(np.clip(1.0 - susceptible.sum() / max(total, 1e-9), 0.0, 1.0))


def run_pipeline(
    panel: pd.DataFrame,
    config: EngineConfig | None = None,
    *,
    planning_interval_months: float = 3.0,
    validation_rounds: int = 2,
    require_trough: bool = False,
    reach_draws_per_unit: int = 400,
) -> PlanSet:
    """Run the whole engine over a panel.

    Args:
        panel: Panel satisfying the data contract.
        config: Engine configuration. Defaults are the programme values.
        planning_interval_months: Months between the rounds being planned. This
            is the tempo of the plan, which is usually faster than the historical
            tempo during an outbreak response.
        validation_rounds: Trailing rounds held out to score the reach model.
        require_trough: Require immunity to hold above the threshold between
            rounds, not only immediately after one.
        reach_draws_per_unit: Draws taken from the predictive reach distribution
            of each unit's next round.

    Returns:
        A ``PlanSet``.
    """
    config = config or EngineConfig()
    validate_panel(panel)

    log = ProvenanceLog()
    features, log = build_features(panel, config, log)

    rng = np.random.default_rng(config.random_seed)

    model = ReachModel()
    reach_report: ReachModelReport | None
    try:
        reach_report = model.fit(features, validation_rounds=validation_rounds)
        fitted = True
    except ValueError as exc:
        reach_report = None
        fitted = False
        log.record(
            "reach_model",
            f"Reach model not fitted: {exc}",
            "Reach falls back to each unit's own recent history, with a wider interval.",
        )

    parameters, notes = estimate_unit_parameters(features, panel, config, log)
    immunity = estimate_current_immunity(features, parameters, panel, config)
    parameters = parameters.join(immunity)

    latest = features.sort_values("round_index").groupby("unit_id").tail(1).set_index("unit_id")
    latest = latest.reindex(parameters.index)

    if fitted:
        draws = model.sample_reach(latest, reach_draws_per_unit, rng)
    else:
        # Without a fitted model, use the unit's own last three rounds and widen
        # by the spread between them. Weak, and labelled as such.
        recent = (
            features.sort_values("round_index").groupby("unit_id")["reach"].apply(
                lambda s: s.tail(3).to_numpy()
            )
        ).reindex(parameters.index)
        draws = np.empty((len(parameters), reach_draws_per_unit))
        for i, values in enumerate(recent):
            values = np.asarray([v for v in (values if values is not None else []) if np.isfinite(v)])
            centre = float(values.mean()) if values.size else 0.6
            spread = float(values.std()) if values.size > 1 else 0.15
            draws[i] = np.clip(rng.normal(centre, max(spread, 0.08), reach_draws_per_unit), 0.0, 1.0)

    plans: list[UnitPlan] = []
    for i, unit in enumerate(parameters.index):
        row = parameters.loc[unit]
        inputs = UnitInputs(
            unit_id=str(unit),
            grain=str(row["grain"]),
            lga_code=str(row["lga_code"]),
            state_code=str(row["state_code"]),
            target_pop=float(row["target_pop"]),
            denominator_basis=str(row["denominator_basis"]),
            reach_draws=draws[i],
            current_immunity=float(row["current_immunity"]),
            current_immunity_sd=float(row["current_immunity_sd"]),
            pi_zero=float(row["pi_zero"]),
            kappa=float(row["kappa"]),
            rho=float(row["rho"]),
            births_per_month=float(row["births_per_month"]),
            ri_protection=float(row["ri_protection"]),
            interval_months=planning_interval_months,
            denominator_inflation_mean=float(row["denominator_inflation"]),
            reach_drift_per_round=float(row["reach_drift_per_round"]),
            reach_wobble_cv=float(row["reach_wobble_cv"]),
        )
        plans.append(solve_unit(inputs, config, rng=rng, require_trough=require_trough))

    notes.append(
        f"Plans assume rounds every {planning_interval_months:.0f} months against a target of "
        f"{config.vc_adjusted:.1%} (R0 {config.r0}, schedule failure {config.schedule_failure})."
    )
    if not fitted:
        notes.append("Reach model was not fitted. Round counts rest on recent history alone.")

    return PlanSet(
        plans=plans,
        reach_report=reach_report,
        reach_model=model if fitted else None,
        provenance=log,
        parameters=parameters,
        notes=notes,
    )
