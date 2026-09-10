"""Susceptible-stock dynamics: what a campaign actually moves.

A campaign delivers doses. Doses are a flow. Population immunity is a stock.
Confusing the two is the denominator paradox that the Kano rounds exposed: teams
report complete household coverage while the target stays unmet, because the
stock is being refilled by births faster than the flow drains it.

This module tracks the stock directly. The susceptible children of a unit are
held as a mass spread over reach propensity, so a campaign pulse removes the
children a team can actually reach and leaves the rest, and newborns enter with
the population's propensity rather than the survivors' propensity.

Two answers come out of it, and they are different questions:

* **Rounds to reach the target.** How many rounds until immunity first crosses
  the threshold.
* **Interval to hold the target.** How often rounds must repeat, for good, so
  that immunity never falls back through the threshold between them. A
  programme can win the first and still lose the second.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .heterogeneity import ReachDistribution


@dataclass(frozen=True)
class PopulationState:
    """Children in the target age band of one unit, resolved by reach propensity.

    Attributes:
        total_children: Size of the target cohort.
        susceptible: Susceptible mass at each reach node. Sums to the number of
            susceptible children.
        nodes: Reach propensity at each node. Node 0 is the unreachable core.
        weights: Share of the *population* at each node. Newborns enter here.
    """

    total_children: float
    susceptible: np.ndarray
    nodes: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.susceptible.shape != self.nodes.shape != self.weights.shape:
            raise ValueError("susceptible, nodes and weights must have the same shape.")
        if np.any(self.susceptible < -1e-9):
            raise ValueError("susceptible mass must not be negative.")
        if self.total_children <= 0:
            raise ValueError("total_children must be positive.")

    @property
    def n_susceptible(self) -> float:
        return float(self.susceptible.sum())

    @property
    def population_immunity(self) -> float:
        """Share of the target cohort that is immune. This is P_eff."""
        return float(np.clip(1.0 - self.n_susceptible / self.total_children, 0.0, 1.0))


def initial_state(
    *,
    total_children: float,
    population_immunity: float,
    reach: ReachDistribution,
    n_nodes: int = 24,
    susceptible_skew: float = 1.0,
) -> PopulationState:
    """Build a starting state from a headline immunity figure.

    Args:
        total_children: Size of the target cohort.
        population_immunity: Current immune share, in ``[0, 1]``.
        reach: Reach distribution of the unit.
        n_nodes: Quadrature resolution.
        susceptible_skew: How much the surviving susceptibles are concentrated in
            hard-to-reach children. ``1.0`` spreads them like the population,
            which is the neutral assumption. Values above 1 tilt them toward low
            reach, which is what past rounds actually do; ``state_from_history``
            derives the tilt instead of assuming it.

    Returns:
        A ``PopulationState``.
    """
    if not 0.0 <= population_immunity <= 1.0:
        raise ValueError("population_immunity must lie in [0, 1].")
    if susceptible_skew <= 0.0:
        raise ValueError("susceptible_skew must be positive.")

    nodes, weights = reach.nodes_and_weights(n_nodes)
    n_susceptible = total_children * (1.0 - population_immunity)

    # Tilt susceptibility toward low-reach children by re-weighting with
    # (1 - r)^(skew - 1), then renormalising.
    tilt = weights * (1.0 - nodes) ** (susceptible_skew - 1.0)
    tilt = tilt / tilt.sum() if tilt.sum() > 0 else weights

    return PopulationState(
        total_children=float(total_children),
        susceptible=n_susceptible * tilt,
        nodes=nodes,
        weights=weights,
    )


def apply_round(state: PopulationState, *, take: float, reach: ReachDistribution) -> PopulationState:
    """Apply one campaign round to the state.

    A child at reach node ``r`` is immunised with probability ``take * r``. After
    the pulse a share ``1 - rho`` of the remaining susceptibles redraw their
    propensity, which is how partial stickiness enters the projection.

    Args:
        state: State before the round.
        take: Per-dose probability that a delivered dose immunises a susceptible child.
        reach: Reach distribution supplying the stickiness ``rho``.

    Returns:
        The state after the round. The cohort size is unchanged; only the split
        between susceptible and immune moves.
    """
    if not 0.0 < take <= 1.0:
        raise ValueError("take must lie in (0, 1].")

    survived = state.susceptible * (1.0 - take * state.nodes)
    total = float(survived.sum())
    if total > 0.0:
        remixed = reach.rho * survived + (1.0 - reach.rho) * total * state.weights
    else:
        remixed = survived
    return replace(state, susceptible=remixed)


def evolve(
    state: PopulationState,
    *,
    months: float,
    births_per_month: float,
    ri_protection: float,
    age_band_months: int = 60,
) -> PopulationState:
    """Age the cohort forward, letting births refill the susceptible stock.

    The target cohort is a stock with births flowing in and children ageing out
    at rate ``1 / age_band_months``. Over an interval of ``months`` the linear
    system has a closed form, so no numerical integration is needed:

        N(t) = N0 * e^(-t/A) + B * A * (1 - e^(-t/A))
        S(t) = S0 * e^(-t/A) + B * (1 - v) * w * A * (1 - e^(-t/A))

    where ``v`` is the share of newborns that routine immunisation protects and
    ``w`` is the population reach distribution that newborns are drawn from.

    Args:
        state: State at the start of the interval.
        months: Length of the interval.
        births_per_month: Births entering the cohort each month.
        ri_protection: Share of newborns protected by routine immunisation, in ``[0, 1]``.
        age_band_months: Width of the target age band.

    Returns:
        The state at the end of the interval.
    """
    if months < 0:
        raise ValueError("months must not be negative.")
    if not 0.0 <= ri_protection <= 1.0:
        raise ValueError("ri_protection must lie in [0, 1].")
    if months == 0:
        return state

    decay = float(np.exp(-months / age_band_months))
    inflow_factor = age_band_months * (1.0 - decay)

    total = state.total_children * decay + births_per_month * inflow_factor
    susceptible = (
        state.susceptible * decay
        + births_per_month * (1.0 - ri_protection) * state.weights * inflow_factor
    )
    return replace(state, total_children=max(total, 1e-9), susceptible=susceptible)


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectory:
    """Immunity over a planned sequence of rounds.

    Attributes:
        immunity_after_round: Population immunity immediately after each round.
            Index 0 is the state before any round.
        immunity_before_next: Population immunity just before the following round,
            after births have refilled the stock. This is the trough that decides
            whether transmission restarts between rounds.
        n_susceptible_after: Susceptible children remaining after each round.
    """

    immunity_after_round: np.ndarray
    immunity_before_next: np.ndarray
    n_susceptible_after: np.ndarray


def project_trajectory(
    state: PopulationState,
    *,
    n_rounds: int,
    take: float,
    reach: ReachDistribution,
    interval_months: float,
    births_per_month: float,
    ri_protection: float,
    age_band_months: int = 60,
) -> Trajectory:
    """Project immunity through a sequence of evenly spaced rounds.

    Args:
        state: Starting state.
        n_rounds: Number of rounds to project.
        take: Per-dose take.
        reach: Reach distribution.
        interval_months: Months between consecutive rounds.
        births_per_month: Births entering the cohort each month.
        ri_protection: Share of newborns protected by routine immunisation.
        age_band_months: Width of the target age band.

    Returns:
        A ``Trajectory`` of length ``n_rounds + 1``.
    """
    if n_rounds < 0:
        raise ValueError("n_rounds must not be negative.")
    if interval_months <= 0:
        raise ValueError("interval_months must be positive.")

    after = np.empty(n_rounds + 1)
    before_next = np.empty(n_rounds + 1)
    susceptible = np.empty(n_rounds + 1)

    current = state
    after[0] = current.population_immunity
    susceptible[0] = current.n_susceptible
    lapsed = evolve(
        current,
        months=interval_months,
        births_per_month=births_per_month,
        ri_protection=ri_protection,
        age_band_months=age_band_months,
    )
    before_next[0] = lapsed.population_immunity

    for i in range(1, n_rounds + 1):
        current = evolve(
            current,
            months=interval_months,
            births_per_month=births_per_month,
            ri_protection=ri_protection,
            age_band_months=age_band_months,
        )
        current = apply_round(current, take=take, reach=reach)
        after[i] = current.population_immunity
        susceptible[i] = current.n_susceptible
        lapsed = evolve(
            current,
            months=interval_months,
            births_per_month=births_per_month,
            ri_protection=ri_protection,
            age_band_months=age_band_months,
        )
        before_next[i] = lapsed.population_immunity

    return Trajectory(
        immunity_after_round=after,
        immunity_before_next=before_next,
        n_susceptible_after=susceptible,
    )


def steady_state_immunity(
    *,
    take: float,
    reach: ReachDistribution,
    interval_months: float,
    births_per_month: float,
    ri_protection: float,
    total_children: float,
    age_band_months: int = 60,
    n_nodes: int = 24,
    max_iterations: int = 400,
    tolerance: float = 1e-7,
) -> tuple[float, float]:
    """Immunity a repeating round schedule settles at, as a peak and a trough.

    Rounds forever at a fixed interval do not drive immunity to 1. Births refill
    the susceptible stock between every pulse, and the unreachable core is never
    drained at all, so the system converges to a sawtooth. The peak is immunity
    just after a round; the trough is immunity just before the next one.

    The trough is the number that matters. A programme whose peak clears the
    herd-immunity threshold but whose trough does not has bought a window, not
    control.

    Args:
        take: Per-dose take.
        reach: Reach distribution.
        interval_months: Months between rounds.
        births_per_month: Births entering the cohort each month.
        ri_protection: Share of newborns protected by routine immunisation.
        total_children: Size of the target cohort, used to seed the iteration.
        age_band_months: Width of the target age band.
        n_nodes: Quadrature resolution.
        max_iterations: Cap on fixed-point iterations.
        tolerance: Convergence tolerance on the susceptible mass.

    Returns:
        A pair ``(peak_immunity, trough_immunity)``.
    """
    state = initial_state(
        total_children=total_children,
        population_immunity=0.0,
        reach=reach,
        n_nodes=n_nodes,
    )
    previous = np.inf
    for _ in range(max_iterations):
        state = evolve(
            state,
            months=interval_months,
            births_per_month=births_per_month,
            ri_protection=ri_protection,
            age_band_months=age_band_months,
        )
        state = apply_round(state, take=take, reach=reach)
        current = state.n_susceptible / state.total_children
        if abs(current - previous) < tolerance:
            break
        previous = current

    peak = state.population_immunity
    trough = evolve(
        state,
        months=interval_months,
        births_per_month=births_per_month,
        ri_protection=ri_protection,
        age_band_months=age_band_months,
    ).population_immunity
    return peak, trough


def holding_interval(
    *,
    target_immunity: float,
    take: float,
    reach: ReachDistribution,
    births_per_month: float,
    ri_protection: float,
    total_children: float,
    age_band_months: int = 60,
    candidate_intervals: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 18.0, 24.0),
) -> float | None:
    """Longest round interval whose steady-state trough still clears the target.

    Args:
        target_immunity: Immunity threshold to hold, in ``[0, 1]``.
        take: Per-dose take.
        reach: Reach distribution.
        births_per_month: Births entering the cohort each month.
        ri_protection: Share of newborns protected by routine immunisation.
        total_children: Size of the target cohort.
        age_band_months: Width of the target age band.
        candidate_intervals: Intervals to test, in months, ascending.

    Returns:
        The longest interval that holds the target, or ``None`` when even the
        shortest tested interval cannot hold it. ``None`` means campaigns at any
        practical frequency will not maintain immunity, and the binding
        constraint is reach or routine immunisation, not round count.
    """
    holds: float | None = None
    for interval in candidate_intervals:
        _, trough = steady_state_immunity(
            take=take,
            reach=reach,
            interval_months=interval,
            births_per_month=births_per_month,
            ri_protection=ri_protection,
            total_children=total_children,
            age_band_months=age_band_months,
        )
        if trough >= target_immunity:
            holds = interval
    return holds
