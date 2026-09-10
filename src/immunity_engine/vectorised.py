"""Batched susceptible-stock simulation across Monte Carlo draws.

The scalar model in ``immunity`` is the readable reference. This module is the
same model written so that every draw advances at once, which is what makes a
settlement-grain run finish. A country has of the order of a hundred thousand
settlements and each needs thousands of draws; a per-draw Python loop turns that
into days.

The one modelling change is how the reach distribution is discretised. The
scalar model uses Gauss-Jacobi nodes, which are exact but sit at different
places for every draw. Here the grid of reach cells is shared and only the cell
*masses* differ, taken from the Beta cumulative distribution so the masses stay
exact even when the density is unbounded at an endpoint. Each cell is
represented by its own conditional mean rather than its midpoint, which keeps
the first moment exact as well. ``tests/test_vectorised.py`` holds the two
implementations to within a small tolerance.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betainc


def _cell_masses_and_means(
    a: np.ndarray, b: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Beta mass in each cell and the conditional mean of reach inside it.

    Args:
        a: First Beta shape per draw, shape ``(n,)``.
        b: Second Beta shape per draw, shape ``(n,)``.
        edges: Ascending cell edges spanning ``[0, 1]``, shape ``(k + 1,)``.

    Returns:
        A pair of arrays of shape ``(n, k)``: the mass in each cell and the mean
        reach of the children in it.
    """
    a_col, b_col = a[:, None], b[:, None]
    edge_row = edges[None, :]

    cdf = betainc(a_col, b_col, edge_row)
    mass = np.diff(cdf, axis=1)

    # E[r * 1{r in cell}] = (a / (a + b)) * [I(a+1, b; hi) - I(a+1, b; lo)].
    partial = np.diff(betainc(a_col + 1.0, b_col, edge_row), axis=1)
    first_moment = (a_col / (a_col + b_col)) * partial

    midpoints = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(mass > 1e-12, first_moment / np.maximum(mass, 1e-300), midpoints[None, :])

    mass = np.clip(mass, 0.0, None)
    total = mass.sum(axis=1, keepdims=True)
    mass = np.divide(mass, total, out=np.full_like(mass, 1.0 / mass.shape[1]), where=total > 0)
    return mass, np.clip(means, 0.0, 1.0)


def build_reach_grid(
    mu: np.ndarray,
    kappa: np.ndarray,
    pi_zero: np.ndarray,
    *,
    n_cells: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretise the reach distribution of every draw onto a shared cell grid.

    Args:
        mu: Mean reach among reachable children, per draw.
        kappa: Beta concentration, per draw.
        pi_zero: Share of children in the unreachable core, per draw.
        n_cells: Number of Beta cells. The unreachable core adds one more node.

    Returns:
        A pair ``(nodes, weights)`` of shape ``(n, n_cells + 1)``. Column 0 is
        the unreachable core at reach 0; the rest carry the Beta cells. Weights
        sum to 1 along each row.
    """
    mu = np.clip(np.asarray(mu, dtype=float), 1e-4, 1.0 - 1e-4)
    kappa = np.clip(np.asarray(kappa, dtype=float), 2.0 + 1e-6, None)
    pi_zero = np.clip(np.asarray(pi_zero, dtype=float), 0.0, 0.999)

    a, b = mu * kappa, (1.0 - mu) * kappa
    edges = np.linspace(0.0, 1.0, n_cells + 1)
    mass, means = _cell_masses_and_means(a, b, edges)

    nodes = np.concatenate([np.zeros((len(mu), 1)), means], axis=1)
    weights = np.concatenate([pi_zero[:, None], (1.0 - pi_zero)[:, None] * mass], axis=1)
    return nodes, weights


def simulate_rounds(
    *,
    mu: np.ndarray,
    kappa: np.ndarray,
    pi_zero: np.ndarray,
    rho: np.ndarray,
    take: np.ndarray,
    initial_immunity: np.ndarray,
    cohort: np.ndarray,
    births_per_month: np.ndarray,
    ri_protection: float,
    interval_months: float,
    target_immunity: float,
    max_rounds: int,
    age_band_months: int = 60,
    require_trough: bool = False,
    n_cells: int = 48,
    reach_drift_per_round: float = 0.0,
    drift_floor: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Run every draw forward and record when each first clears the target.

    Args:
        mu: Mean reach among reachable children, per draw.
        kappa: Beta concentration, per draw.
        pi_zero: Unreachable core share, per draw.
        rho: Round-to-round stickiness, per draw.
        take: Per-dose take, per draw.
        initial_immunity: Population immunity now, per draw.
        cohort: True target cohort size, per draw.
        births_per_month: Births entering the cohort each month, per draw.
        ri_protection: Share of newborns protected by routine immunisation.
        interval_months: Months between rounds.
        target_immunity: Threshold to clear.
        max_rounds: Round budget.
        age_band_months: Width of the target age band.
        require_trough: Require immunity to still clear the target just before
            the next round, not only just after this one.
        n_cells: Reach grid resolution.
        reach_drift_per_round: Proportional change in reach with each successive
            round. Negative values represent team and community fatigue, which is
            measurable in the panel and is otherwise invisible to a model that
            assumes reach is stationary.
        drift_floor: Lower bound on the cumulative drift factor. Fatigue plateaus;
            it does not compound to zero, and letting it do so would manufacture
            infeasibility.

    Returns:
        A pair ``(rounds_needed, immunity_after)``. ``rounds_needed`` holds the
        first clearing round per draw, or ``inf`` where the budget never clears
        it. ``immunity_after`` has shape ``(n, max_rounds + 1)`` and holds
        immunity after each round, so trajectories can be plotted or audited.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1.")
    if interval_months <= 0:
        raise ValueError("interval_months must be positive.")

    nodes, weights = build_reach_grid(mu, kappa, pi_zero, n_cells=n_cells)
    n = nodes.shape[0]

    cohort = np.maximum(np.asarray(cohort, dtype=float), 1.0)
    births = np.maximum(np.asarray(births_per_month, dtype=float), 0.0)
    take_col = np.asarray(take, dtype=float)[:, None]
    rho_col = np.asarray(rho, dtype=float)[:, None]

    susceptible = (cohort * (1.0 - np.asarray(initial_immunity, dtype=float)))[:, None] * weights
    total = cohort.copy()

    decay = float(np.exp(-interval_months / age_band_months))
    inflow = age_band_months * (1.0 - decay)
    susceptible_inflow = (births * (1.0 - ri_protection) * inflow)[:, None] * weights
    total_inflow = births * inflow

    rounds_needed = np.full(n, np.inf)
    immunity_after = np.empty((n, max_rounds + 1))
    immunity_after[:, 0] = np.clip(1.0 - susceptible.sum(axis=1) / total, 0.0, 1.0)

    for r in range(1, max_rounds + 1):
        # Reach drifts with each successive round, then plateaus at the floor.
        drift = max((1.0 + reach_drift_per_round) ** (r - 1), drift_floor)
        survival = 1.0 - take_col * drift * nodes

        total = total * decay + total_inflow
        susceptible = susceptible * decay + susceptible_inflow

        susceptible = susceptible * survival
        remaining = susceptible.sum(axis=1, keepdims=True)
        susceptible = rho_col * susceptible + (1.0 - rho_col) * remaining * weights

        immunity = np.clip(1.0 - susceptible.sum(axis=1) / total, 0.0, 1.0)
        immunity_after[:, r] = immunity

        if require_trough:
            lapsed_total = total * decay + total_inflow
            lapsed_susceptible = susceptible * decay + susceptible_inflow
            achieved = np.clip(1.0 - lapsed_susceptible.sum(axis=1) / lapsed_total, 0.0, 1.0)
        else:
            achieved = immunity

        newly = (achieved >= target_immunity) & ~np.isfinite(rounds_needed)
        rounds_needed[newly] = r

    return rounds_needed, immunity_after


def steady_state_batch(
    *,
    mu: float,
    kappa: float,
    pi_zero: float,
    rho: float,
    take: float,
    births_per_month: float,
    ri_protection: float,
    interval_months: float,
    cohort: float,
    age_band_months: int = 60,
    n_cells: int = 48,
    max_iterations: int = 400,
    tolerance: float = 1e-8,
) -> tuple[float, float]:
    """Peak and trough immunity of a schedule repeated indefinitely.

    Args:
        mu: Mean reach among reachable children.
        kappa: Beta concentration.
        pi_zero: Unreachable core share.
        rho: Round-to-round stickiness.
        take: Per-dose take.
        births_per_month: Births entering the cohort each month.
        ri_protection: Share of newborns protected by routine immunisation.
        interval_months: Months between rounds.
        cohort: Target cohort size.
        age_band_months: Width of the target age band.
        n_cells: Reach grid resolution.
        max_iterations: Cap on fixed-point iterations.
        tolerance: Convergence tolerance on the susceptible share.

    Returns:
        A pair ``(peak, trough)`` of population immunity.
    """
    nodes, weights = build_reach_grid(
        np.array([mu]), np.array([kappa]), np.array([pi_zero]), n_cells=n_cells
    )
    susceptible = cohort * weights.copy()
    total = float(cohort)

    decay = float(np.exp(-interval_months / age_band_months))
    inflow = age_band_months * (1.0 - decay)
    susceptible_inflow = births_per_month * (1.0 - ri_protection) * inflow * weights
    total_inflow = births_per_month * inflow
    survival = 1.0 - take * nodes

    previous = np.inf
    for _ in range(max_iterations):
        total = total * decay + total_inflow
        susceptible = susceptible * decay + susceptible_inflow
        susceptible = susceptible * survival
        remaining = susceptible.sum()
        susceptible = rho * susceptible + (1.0 - rho) * remaining * weights
        current = float(susceptible.sum() / total)
        if abs(current - previous) < tolerance:
            break
        previous = current

    peak = float(np.clip(1.0 - susceptible.sum() / total, 0.0, 1.0))
    trough_total = total * decay + total_inflow
    trough_susceptible = susceptible * decay + susceptible_inflow
    trough = float(np.clip(1.0 - trough_susceptible.sum() / trough_total, 0.0, 1.0))
    return peak, trough
