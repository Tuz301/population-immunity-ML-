"""The answer: how many rounds, with what confidence, and when the answer is none.

Everything upstream produces distributions. This module turns them into the two
numbers a programme can act on and the one verdict it most needs to hear.

The question "how many campaigns" is inverted rather than regressed. There is no
model that maps features to a round count directly, because the round count is
not a property of a place; it is the answer to a threshold-crossing problem over
a stock that births refill and that a bounded share of children can never be
reached in. Regressing on it would learn the programme's past round schedule and
call it a requirement.

So instead: draw the parameters, run the stock forward, record the round at which
immunity first clears the threshold, and report the distribution of that round.
Where the immunity ceiling sits below the threshold in a draw, no round clears
it, and that draw votes for the verdict that campaigns are not the instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .contracts import EngineConfig, Feasibility
from .heterogeneity import ReachDistribution
from .vectorised import simulate_rounds, steady_state_batch

#: Rounds simulated beyond the budget so the immunity trajectory converges and
#: the ceiling is a property of the schedule rather than of where we stopped.
CEILING_HORIZON_ROUNDS = 60


@dataclass(frozen=True)
class UnitInputs:
    """Everything the inversion needs about one geographic unit.

    Attributes:
        unit_id: Identifier of the unit.
        grain: Settlement, ward or LGA.
        lga_code: LGA the unit sits in.
        state_code: State the unit sits in.
        target_pop: Target cohort size, as reported.
        denominator_basis: Provenance stamp on ``target_pop``.
        reach_draws: Draws from the predicted per-round reach distribution.
        current_immunity: Current population immunity, in ``[0, 1]``.
        current_immunity_sd: Uncertainty on that estimate.
        pi_zero: Share of children with zero reach in every round.
        kappa: Beta concentration of reach across children.
        rho: Round-to-round stickiness of reach.
        births_per_month: Births entering the cohort each month.
        ri_protection: Share of newborns protected by routine immunisation.
        interval_months: Planned months between rounds.
        r0_local: Local reproduction number, if estimated.
        denominator_inflation_mean: Factor by which ``target_pop`` overstates the
            true cohort. 1.0 means the denominator is believed.
        denominator_inflation_sd: Uncertainty on that factor.
        reach_drift_per_round: Measured proportional change in reach with each
            successive round. Negative is fatigue.
    """

    unit_id: str
    grain: str
    lga_code: str
    state_code: str
    target_pop: float
    denominator_basis: str
    reach_draws: np.ndarray
    current_immunity: float
    current_immunity_sd: float
    pi_zero: float
    kappa: float
    rho: float
    births_per_month: float
    ri_protection: float
    interval_months: float = 6.0
    r0_local: float | None = None
    denominator_inflation_mean: float = 1.0
    denominator_inflation_sd: float = 0.08
    reach_drift_per_round: float = 0.0


@dataclass(frozen=True)
class UnitPlan:
    """The campaign requirement for one unit.

    Attributes:
        unit_id: Identifier of the unit.
        grain: Settlement, ward or LGA.
        lga_code: LGA the unit sits in.
        state_code: State the unit sits in.
        target_immunity: Immunity threshold used, in ``[0, 1]``.
        current_immunity: Estimated immunity now.
        feasibility: Verdict.
        rounds_median: Rounds needed in the median draw. ``None`` when the median
            draw never clears the threshold.
        rounds_assured: Rounds needed to clear the threshold in ``assurance`` of
            draws. This is the planning number.
        rounds_interval_80: Central 80% interval on the round count.
        probability_by_round: For each round count, the share of draws in which
            that many rounds clears the threshold. The full answer, before it is
            summarised into a single number.
        immunity_ceiling_median: Highest immunity this schedule ever reaches, once
            repeated rounds and births balance. Births refill the susceptible
            stock between rounds, so this sits below the core-limited ceiling and
            it is the one that decides feasibility.
        core_ceiling_median: Highest immunity a closed cohort could reach, limited
            only by the unreachable core. Reported alongside because the gap
            between the two ceilings separates an access problem from a
            round-frequency problem.
        immunity_trough_median: Immunity between rounds once the schedule settles.
        holding_interval_months: Longest interval that holds the threshold, or
            ``None`` when no tested interval holds it.
        probability_infeasible: Share of draws in which the ceiling sits below the
            threshold even with reach held at its current level. This is the
            structural verdict: it does not depend on any assumption about future
            decline.
        fatigue_risk: True when the target is reachable at today's reach but not
            at the rate reach is currently falling. A different problem from a
            structural ceiling, with a different owner: it is about holding
            performance, not about access.
        probability_infeasible_with_drift: Share of draws that fall short once the
            measured decline in reach is projected forward.
        binding_constraint: The input whose uncertainty drives the round count.
        interval_months: Months between rounds in the plan that was evaluated.
        n_draws: Draws behind the figures.
    """

    unit_id: str
    grain: str
    lga_code: str
    state_code: str
    target_immunity: float
    current_immunity: float
    feasibility: Feasibility
    rounds_median: int | None
    rounds_assured: int | None
    rounds_interval_80: tuple[int | None, int | None]
    probability_by_round: dict[int, float]
    immunity_ceiling_median: float
    core_ceiling_median: float
    immunity_trough_median: float
    holding_interval_months: float | None
    probability_infeasible: float
    fatigue_risk: bool
    probability_infeasible_with_drift: float
    binding_constraint: str
    interval_months: float
    n_draws: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feasibility"] = self.feasibility.value
        return payload

    def headline(self) -> str:
        """One sentence a programme manager can act on."""
        if self.feasibility is Feasibility.INFEASIBLE_BY_CAMPAIGNS:
            core_binds = self.core_ceiling_median < self.target_immunity + 0.005
            cause = (
                "an unreachable core of children that no round touches"
                if core_binds
                else f"births refilling the susceptible stock faster than rounds every "
                     f"{self.interval_months:.0f} months drain it"
            )
            remedy = (
                "Negotiate access, enumerate the mobile population, or reconcile the denominator."
                if core_binds
                else (
                    f"Shorten the interval to {self.holding_interval_months:.0f} months"
                    if self.holding_interval_months
                    else "Raise reach or routine immunisation"
                ) + "; more rounds at this spacing will not close the gap."
            )
            return (
                f"{self.unit_id}: no number of rounds at this schedule reaches "
                f"{self.target_immunity:.1%}. The ceiling is {self.immunity_ceiling_median:.1%}, "
                f"held down by {cause} in {self.probability_infeasible:.0%} of draws. {remedy}"
            )
        if self.feasibility is Feasibility.INSUFFICIENT_DATA:
            return f"{self.unit_id}: inputs too thin for a round count. No number is offered."
        if self.feasibility is Feasibility.FEASIBLE_BEYOND_BUDGET:
            return (
                f"{self.unit_id}: more than {max(self.probability_by_round)} rounds are needed "
                f"to reach {self.target_immunity:.1%}. Within budget the plan does not get there."
            )
        if self.fatigue_risk:
            return (
                f"{self.unit_id}: {self.rounds_assured} rounds reach "
                f"{self.target_immunity:.1%} only if reach is held at today's level. At the "
                f"rate reach is currently falling, the target is missed. Hold performance "
                f"first; the round count is contingent on it."
            )
        return (
            f"{self.unit_id}: {self.rounds_assured} rounds reach {self.target_immunity:.1%} "
            f"with {self.probability_by_round.get(self.rounds_assured, 0.0):.0%} confidence "
            f"(median {self.rounds_median}, 80% interval "
            f"{self.rounds_interval_80[0]} to {self.rounds_interval_80[1]})."
        )


def _draw_parameters(
    inputs: UnitInputs, config: EngineConfig, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Draw the uncertain parameters once per Monte Carlo replicate."""
    n = config.n_draws

    reach = inputs.reach_draws
    if reach.size == 0:
        raise ValueError(f"Unit {inputs.unit_id} has no reach draws.")
    reach_draws = rng.choice(reach.ravel(), size=n, replace=True)

    take = np.clip(
        rng.normal(config.per_dose_take_mean, config.per_dose_take_sd, size=n), 0.05, 1.0
    )
    immunity = np.clip(
        rng.normal(inputs.current_immunity, max(inputs.current_immunity_sd, 1e-6), size=n),
        0.0,
        0.999,
    )
    # Denominator inflation is lognormal: it is a multiplicative error that
    # cannot go negative and is right-skewed, which is how the Kano-type
    # inflation actually presents.
    sigma = np.sqrt(np.log1p((inputs.denominator_inflation_sd / max(inputs.denominator_inflation_mean, 1e-6)) ** 2))
    mu = np.log(max(inputs.denominator_inflation_mean, 1e-6)) - 0.5 * sigma**2
    inflation = np.clip(rng.lognormal(mu, sigma, size=n), 0.5, 3.0)

    # Stickiness and concentration carry their own estimation error.
    rho = np.clip(rng.normal(inputs.rho, 0.10, size=n), 0.0, 0.98)
    kappa = np.clip(rng.normal(inputs.kappa, 0.25 * inputs.kappa, size=n), 2.5, 200.0)
    pi_zero = np.clip(rng.normal(inputs.pi_zero, 0.30 * inputs.pi_zero + 0.005, size=n), 0.0, 0.60)

    return {
        "reach": reach_draws,
        "take": take,
        "immunity": immunity,
        "inflation": inflation,
        "rho": rho,
        "kappa": kappa,
        "pi_zero": pi_zero,
    }


def _binding_constraint(
    rounds: np.ndarray, drawn: dict[str, np.ndarray], max_rounds: int
) -> str:
    """Name the input whose variation moves the round count most.

    Reported so that the next data-collection pound is spent where it shortens
    the interval, rather than on whatever is easiest to measure.
    """
    finite = np.isfinite(rounds)
    if finite.sum() < 30 or np.nanstd(rounds[finite]) < 1e-9:
        return "not identifiable from this sample"

    labels = {
        "reach": "per-round reach (invest in verified coverage measurement)",
        "take": "per-dose vaccine take (invest in cold chain and serology)",
        "immunity": "current immunity estimate (invest in a serosurvey or LQAS)",
        "inflation": "denominator provenance (invest in microplan reconciliation)",
        "pi_zero": "size of the unreachable core (invest in access negotiation and enumeration)",
        "rho": "chronic-miss stickiness (invest in tracking named missed children)",
        "kappa": "spread of reach across children (invest in settlement-level verification)",
    }
    scores: dict[str, float] = {}
    y = rounds[finite]
    for name, values in drawn.items():
        x = values[finite]
        if np.std(x) < 1e-12:
            continue
        scores[name] = abs(float(np.corrcoef(x, y)[0, 1]))
    if not scores:
        return "not identifiable from this sample"
    winner = max(scores, key=scores.get)
    return f"{labels.get(winner, winner)} — rank correlation {scores[winner]:.2f} with round count"


def solve_unit(
    inputs: UnitInputs,
    config: EngineConfig,
    *,
    rng: np.random.Generator | None = None,
    require_trough: bool = False,
) -> UnitPlan:
    """Compute the campaign requirement for one unit.

    Args:
        inputs: Unit inputs.
        config: Engine configuration.
        rng: Random generator. Created from ``config.random_seed`` when omitted.
        require_trough: When True, a round only counts as clearing the threshold
            if immunity is still above it just before the *next* round. This is
            the stricter and more honest criterion for interrupting transmission,
            and it usually costs one to two extra rounds.

    Returns:
        A ``UnitPlan``.
    """
    rng = rng if rng is not None else np.random.default_rng(config.random_seed)

    target = (
        config.vc_for_r0(inputs.r0_local)
        if (config.use_local_r0 and inputs.r0_local)
        else config.vc_adjusted
    )
    target = float(np.clip(target, 0.0, 0.999))

    if inputs.target_pop <= 0 or not np.isfinite(inputs.current_immunity):
        return _insufficient(inputs, target, config)

    drawn = _draw_parameters(inputs, config, rng)

    # Reach from the model is a population average that already contains the
    # unreachable core. The Beta component describes only the reachable children,
    # so the core is divided out before it is used as a Beta mean.
    pi_zero = drawn["pi_zero"]
    mu = np.clip(drawn["reach"] / np.maximum(1.0 - pi_zero, 1e-6), 1e-4, 1.0 - 1e-4)
    cohort = np.maximum(inputs.target_pop / drawn["inflation"], 1.0)

    # Run past the round budget so the trajectory converges. The converged
    # immunity is the ceiling this schedule can ever reach; the first crossing
    # inside the budget is the round count. One pass gives both.
    horizon = max(config.max_rounds, CEILING_HORIZON_ROUNDS)
    simulation_kwargs = dict(
        mu=mu,
        kappa=drawn["kappa"],
        pi_zero=pi_zero,
        rho=drawn["rho"],
        take=drawn["take"],
        initial_immunity=drawn["immunity"],
        cohort=cohort,
        births_per_month=inputs.births_per_month / drawn["inflation"],
        ri_protection=inputs.ri_protection,
        interval_months=inputs.interval_months,
        target_immunity=target,
        max_rounds=horizon,
        age_band_months=config.target_age_months,
        require_trough=require_trough,
    )

    # Two runs, because two different questions are being asked.
    #
    # The planning run projects the decline in reach that the panel actually
    # shows, so the round count is what this programme should expect.
    #
    # The structural run holds reach at today's level. Its ceiling answers
    # whether the target is reachable at all, independently of any assumption
    # about future performance. Only that run may return the verdict that
    # campaigns are the wrong instrument, because a shortfall caused by teams
    # tiring is a shortfall someone can do something about, and telling a
    # programme its access is hopeless when its problem is fatigue sends the
    # response to the wrong place entirely.
    rounds_to_cross, immunity_path = simulate_rounds(
        reach_drift_per_round=inputs.reach_drift_per_round,
        drift_floor=config.reach_drift_floor,
        **simulation_kwargs,
    )
    _, stationary_path = simulate_rounds(reach_drift_per_round=0.0, **simulation_kwargs)

    # Crossings beyond the budget are not plans, so they are censored back to
    # infinity for the round-count summaries.
    rounds_needed = np.where(rounds_to_cross <= config.max_rounds, rounds_to_cross, np.inf)

    schedule_ceiling = stationary_path[:, -1]
    drifted_ceiling = immunity_path[:, -1]
    core_ceiling = 1.0 - (1.0 - drawn["immunity"]) * pi_zero
    ceilings = schedule_ceiling

    finite = np.isfinite(rounds_needed)
    share_reached = float(finite.mean())
    probability_by_round = {
        r: float(np.mean(rounds_needed <= r)) for r in range(1, config.max_rounds + 1)
    }

    ceiling_median = float(np.median(ceilings))
    core_ceiling_median = float(np.median(core_ceiling))
    probability_infeasible = float(np.mean(ceilings < target))
    probability_infeasible_with_drift = float(np.mean(drifted_ceiling < target))
    fatigue_risk = bool(
        probability_infeasible < 0.50 <= probability_infeasible_with_drift
    )

    # The verdict. Order matters: an unreachable core is a different problem
    # from a round budget that is merely too small, and must not be reported
    # as the same thing.
    if probability_infeasible >= 0.50:
        feasibility = Feasibility.INFEASIBLE_BY_CAMPAIGNS
    elif share_reached >= config.assurance:
        feasibility = Feasibility.FEASIBLE
    else:
        feasibility = Feasibility.FEASIBLE_BEYOND_BUDGET

    rounds_median = _quantile_round(rounds_needed, 0.50)
    rounds_assured = _quantile_round(rounds_needed, config.assurance)
    interval_80 = (
        _quantile_round(rounds_needed, 0.10),
        _quantile_round(rounds_needed, 0.90),
    )

    mu_point = float(np.clip(np.median(drawn["reach"]) / max(1.0 - inputs.pi_zero, 1e-6), 1e-4, 1.0 - 1e-4))
    steady_kwargs = dict(
        mu=mu_point,
        kappa=inputs.kappa,
        pi_zero=inputs.pi_zero,
        rho=inputs.rho,
        take=config.per_dose_take_mean,
        births_per_month=inputs.births_per_month,
        ri_protection=inputs.ri_protection,
        cohort=max(inputs.target_pop, 1.0),
        age_band_months=config.target_age_months,
    )
    _, trough = steady_state_batch(interval_months=inputs.interval_months, **steady_kwargs)

    # Longest repeating interval whose trough still clears the target. None means
    # no practical schedule holds it, so the constraint is reach, not frequency.
    hold: float | None = None
    for candidate in (2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 18.0, 24.0):
        _, candidate_trough = steady_state_batch(interval_months=candidate, **steady_kwargs)
        if candidate_trough >= target:
            hold = candidate

    return UnitPlan(
        unit_id=inputs.unit_id,
        grain=inputs.grain,
        lga_code=inputs.lga_code,
        state_code=inputs.state_code,
        target_immunity=target,
        current_immunity=float(inputs.current_immunity),
        feasibility=feasibility,
        rounds_median=rounds_median,
        rounds_assured=rounds_assured,
        rounds_interval_80=interval_80,
        probability_by_round=probability_by_round,
        immunity_ceiling_median=ceiling_median,
        core_ceiling_median=core_ceiling_median,
        immunity_trough_median=float(trough),
        holding_interval_months=hold,
        probability_infeasible=probability_infeasible,
        fatigue_risk=fatigue_risk,
        probability_infeasible_with_drift=probability_infeasible_with_drift,
        binding_constraint=_binding_constraint(rounds_needed, drawn, config.max_rounds),
        interval_months=float(inputs.interval_months),
        n_draws=config.n_draws,
    )


def _quantile_round(rounds: np.ndarray, level: float) -> int | None:
    """Quantile of a round count that may be infinite in some draws.

    ``numpy.quantile`` propagates the infinities. Sorting and indexing does not,
    and returning ``None`` above the finite share is the honest answer: at that
    confidence level, no round count suffices.
    """
    ordered = np.sort(rounds)
    index = int(np.ceil(level * len(ordered))) - 1
    index = max(0, min(index, len(ordered) - 1))
    value = ordered[index]
    return None if not np.isfinite(value) else int(value)


def _insufficient(inputs: UnitInputs, target: float, config: EngineConfig) -> UnitPlan:
    """Plan returned when the inputs cannot support a number."""
    return UnitPlan(
        unit_id=inputs.unit_id,
        grain=inputs.grain,
        lga_code=inputs.lga_code,
        state_code=inputs.state_code,
        target_immunity=target,
        current_immunity=float(inputs.current_immunity)
        if np.isfinite(inputs.current_immunity)
        else float("nan"),
        feasibility=Feasibility.INSUFFICIENT_DATA,
        rounds_median=None,
        rounds_assured=None,
        rounds_interval_80=(None, None),
        probability_by_round={},
        immunity_ceiling_median=float("nan"),
        core_ceiling_median=float("nan"),
        immunity_trough_median=float("nan"),
        holding_interval_months=None,
        probability_infeasible=float("nan"),
        fatigue_risk=False,
        probability_infeasible_with_drift=float("nan"),
        binding_constraint="no usable denominator or immunity estimate",
        interval_months=float(inputs.interval_months),
        n_draws=0,
    )
