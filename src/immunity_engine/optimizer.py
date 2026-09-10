"""Where the next round should go, when there are not enough rounds to go round.

The round count per unit answers "what would it take here". A programme also has
to answer "we can fund eleven rounds this year, where do they go", and those are
different questions with different answers. A unit needing two rounds to cross
the threshold may deserve them ahead of a unit needing six, or may not, depending
on how many susceptible children sit behind each round and how hard the virus is
being pushed in each place.

The marginal immunity a round buys falls with every previous round in the same
unit: the easy children are reached first. That makes the objective submodular,
and greedy allocation on a submodular objective is within a factor of
``1 - 1/e`` of the best possible allocation. It is also the only method a
programme can audit line by line, which matters more than the last few percent.

The optimiser will not allocate rounds to a unit whose ceiling sits below the
target. Sending rounds there is the specific mistake the engine exists to
prevent, so it is refused in code rather than discouraged in a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .contracts import EngineConfig, Feasibility
from .rounds_required import UnitPlan
from .vectorised import simulate_rounds


class Objective(str, Enum):
    """What the allocation is trying to maximise."""

    SUSCEPTIBLES_AVERTED = "susceptibles_averted"
    """Risk-weighted susceptible children removed from the stock.

    The default. It respects population size, so a round in a large LGA is not
    treated as equal to a round in a small one.
    """

    UNITS_ABOVE_TARGET = "units_above_target"
    """Number of units brought above the immunity threshold.

    Use when the operational goal is certification-style coverage of units rather
    than transmission reduction. It will starve large, difficult units in favour
    of small, nearly-there ones, which is sometimes right and is always a choice.
    """


@dataclass(frozen=True)
class Allocation:
    """A round budget spread across units.

    Attributes:
        table: One row per unit that received rounds, with the marginal value of
            each round it got.
        rounds_allocated: Total rounds placed.
        objective_value: Value achieved.
        objective_uniform: Value the same budget achieves spread evenly across
            eligible units. The gap is what the optimisation is worth.
        refused: Units excluded because their ceiling sits below the target,
            with the reason. These need a different instrument, not a round.
        notes: Anything a reader must carry with the allocation.
    """

    table: pd.DataFrame
    rounds_allocated: int
    objective_value: float
    objective_uniform: float
    refused: pd.DataFrame
    notes: list[str]

    @property
    def uplift(self) -> float:
        """Share by which the allocation beats spreading the budget evenly."""
        if self.objective_uniform <= 0:
            return float("nan")
        return self.objective_value / self.objective_uniform - 1.0


def _marginal_curve(
    plan: UnitPlan,
    config: EngineConfig,
    *,
    births_per_month: float,
    ri_protection: float,
    cohort: float,
    max_rounds: int,
) -> np.ndarray:
    """Susceptible children removed by each successive round in one unit.

    Returns:
        An array of length ``max_rounds``. Entry ``i`` is the extra susceptible
        children removed by round ``i + 1`` given that ``i`` rounds already ran.
    """
    reach = np.array([max(plan.current_immunity, 0.0)])  # placeholder, replaced below
    # The plan already carries the reach distribution implicitly through its
    # trajectory, so the curve is recomputed from the same primitives at the
    # plan's own central parameters.
    mu = np.array([_central_reach(plan)])
    _, path = simulate_rounds(
        mu=mu,
        kappa=np.array([8.0]),
        pi_zero=np.array([_implied_pi_zero(plan)]),
        rho=np.array([0.7]),
        take=np.array([config.per_dose_take_mean]),
        initial_immunity=np.array([plan.current_immunity]),
        cohort=np.array([cohort]),
        births_per_month=np.array([births_per_month]),
        ri_protection=ri_protection,
        interval_months=plan.interval_months,
        target_immunity=plan.target_immunity,
        max_rounds=max_rounds,
    )
    susceptible = (1.0 - path[0]) * cohort
    return np.maximum(-np.diff(susceptible), 0.0)


def _central_reach(plan: UnitPlan) -> float:
    """Reach implied by the plan's own ceiling and trough. Bounded away from 0 and 1."""
    return float(np.clip(0.5 + 0.5 * plan.immunity_trough_median, 0.05, 0.98))


def _implied_pi_zero(plan: UnitPlan) -> float:
    """Unreachable core implied by the plan's core-limited ceiling."""
    susceptible = max(1.0 - plan.current_immunity, 1e-6)
    return float(np.clip((1.0 - plan.core_ceiling_median) / susceptible, 0.0, 0.6))


def allocate_rounds(
    plans: list[UnitPlan],
    parameters: pd.DataFrame,
    config: EngineConfig,
    *,
    round_budget: int,
    objective: Objective = Objective.SUSCEPTIBLES_AVERTED,
    risk_weight: pd.Series | None = None,
    max_rounds_per_unit: int = 6,
) -> Allocation:
    """Spread a fixed number of rounds across units to maximise the objective.

    Args:
        plans: Plans from ``pipeline.run_pipeline``.
        parameters: Per-unit parameters from the same run, indexed by ``unit_id``.
        config: Engine configuration.
        round_budget: Total rounds available.
        objective: What to maximise.
        risk_weight: Optional per-unit multiplier on the value of a round, indexed
            by ``unit_id``. Use it to carry transmission signal that the immunity
            model does not see, such as recent environmental surveillance
            positives or a local reproduction number above one. Missing units
            default to 1.
        max_rounds_per_unit: Cap on rounds in any one unit. Rounds beyond about
            six in a single unit hit community fatigue that the model does not
            represent, so the cap is a guard against extrapolating past what the
            data support.

    Returns:
        An ``Allocation``.

    Raises:
        ValueError: If the budget is not positive or no unit is eligible.
    """
    if round_budget < 1:
        raise ValueError("round_budget must be at least 1.")

    notes: list[str] = []
    eligible: list[UnitPlan] = []
    refused_rows: list[dict] = []

    for plan in plans:
        if plan.feasibility is Feasibility.INFEASIBLE_BY_CAMPAIGNS:
            refused_rows.append(
                {
                    "unit_id": plan.unit_id,
                    "lga_code": plan.lga_code,
                    "current_immunity": plan.current_immunity,
                    "immunity_ceiling": plan.immunity_ceiling_median,
                    "reason": "Ceiling sits below the target; rounds cannot close the gap.",
                }
            )
        elif plan.feasibility is Feasibility.INSUFFICIENT_DATA:
            refused_rows.append(
                {
                    "unit_id": plan.unit_id,
                    "lga_code": plan.lga_code,
                    "current_immunity": plan.current_immunity,
                    "immunity_ceiling": float("nan"),
                    "reason": "Inputs too thin to value a round here.",
                }
            )
        else:
            eligible.append(plan)

    if not eligible:
        raise ValueError(
            "No unit is eligible for rounds. Every unit is either ceiling-limited or "
            "lacks the data to value a round. The budget should go to access, "
            "enumeration or verification instead."
        )

    weights = (
        risk_weight.reindex([p.unit_id for p in eligible]).fillna(1.0)
        if risk_weight is not None
        else pd.Series(1.0, index=[p.unit_id for p in eligible])
    )

    # Marginal value curve per unit, computed once. Because each unit's curve is
    # decreasing, the greedy choice only ever needs the next entry.
    curves: dict[str, np.ndarray] = {}
    for plan in eligible:
        row = parameters.loc[plan.unit_id] if plan.unit_id in parameters.index else None
        cohort = float(row["target_pop"]) if row is not None else 10000.0
        births = float(row["births_per_month"]) if row is not None else cohort / 60.0
        ri = float(row["ri_protection"]) if row is not None else config.default_ri_coverage
        curve = _marginal_curve(
            plan, config,
            births_per_month=births, ri_protection=ri,
            cohort=cohort, max_rounds=max_rounds_per_unit,
        )
        if objective is Objective.UNITS_ABOVE_TARGET:
            # Value is one unit crossing the threshold, credited to the round
            # that crosses it and nothing thereafter.
            crossing = plan.rounds_median
            curve = np.zeros(max_rounds_per_unit)
            if crossing is not None and 1 <= crossing <= max_rounds_per_unit:
                curve[crossing - 1] = 1.0
        curves[plan.unit_id] = curve * float(weights[plan.unit_id])

    # --- greedy allocation ------------------------------------------------
    allocated = {plan.unit_id: 0 for plan in eligible}
    picks: list[dict] = []
    total_value = 0.0

    for _ in range(round_budget):
        best_unit, best_gain = None, 0.0
        for unit, curve in curves.items():
            used = allocated[unit]
            if used >= max_rounds_per_unit:
                continue
            gain = float(curve[used])
            if gain > best_gain:
                best_unit, best_gain = unit, gain
        if best_unit is None:
            notes.append(
                f"Budget exhausted early: after {sum(allocated.values())} rounds no remaining "
                "round adds value. The rest of the budget belongs elsewhere."
            )
            break
        allocated[best_unit] += 1
        total_value += best_gain
        picks.append(
            {
                "unit_id": best_unit,
                "round_number_in_unit": allocated[best_unit],
                "marginal_value": best_gain,
            }
        )

    # --- what spreading the budget evenly would have achieved --------------
    per_unit = round_budget // max(len(eligible), 1)
    remainder = round_budget - per_unit * len(eligible)
    uniform_value = 0.0
    for i, plan in enumerate(eligible):
        count = min(per_unit + (1 if i < remainder else 0), max_rounds_per_unit)
        uniform_value += float(curves[plan.unit_id][:count].sum())

    by_unit = pd.DataFrame(picks)
    if not by_unit.empty:
        summary = (
            by_unit.groupby("unit_id")
            .agg(rounds=("round_number_in_unit", "max"), value=("marginal_value", "sum"))
            .reset_index()
            .sort_values("value", ascending=False)
        )
        lookup = {p.unit_id: p for p in eligible}
        summary["lga_code"] = summary["unit_id"].map(lambda u: lookup[u].lga_code)
        summary["state_code"] = summary["unit_id"].map(lambda u: lookup[u].state_code)
        summary["current_immunity"] = summary["unit_id"].map(
            lambda u: lookup[u].current_immunity
        )
        summary["rounds_to_target"] = summary["unit_id"].map(lambda u: lookup[u].rounds_median)
    else:
        summary = pd.DataFrame()

    notes.append(
        f"{len(refused_rows)} units were refused rounds because a round cannot move them. "
        "They belong in the access and denominator workstream, not the campaign calendar."
    )
    if objective is Objective.UNITS_ABOVE_TARGET:
        notes.append(
            "Objective counts units crossing the threshold, so it ignores how many children "
            "sit behind each unit. Compare against the susceptibles-averted allocation before "
            "committing."
        )

    return Allocation(
        table=summary,
        rounds_allocated=int(sum(allocated.values())),
        objective_value=total_value,
        objective_uniform=uniform_value,
        refused=pd.DataFrame(refused_rows),
        notes=notes,
    )
