"""Reach heterogeneity: why the tenth campaign does not do what the first did.

The naive model treats every round as an independent coin flip over the same
children, so susceptibles fall geometrically and any target is reached with
enough rounds. Field data contradict that. The same settlements and the same
households are missed round after round, which is what the chronic-miss view
in the NEOC portal measures directly.

This module represents reach as a *distribution over children* rather than a
single rate, and represents the round-to-round stickiness of that distribution
as an explicit, estimable parameter. Both are what make the round count finite
and honest.

Model
-----
A child has a per-round probability ``r`` of being reached by a team.

* With probability ``pi_zero`` the child sits in the unreachable core:
  ``r = 0`` in every round. Security-inaccessible settlements, hard refusals
  and never-enumerated mobile populations live here.
* Otherwise ``r ~ Beta(a, b)`` with mean ``mu`` and concentration ``kappa``.

Between rounds a share ``rho`` of children keep their propensity and a share
``1 - rho`` redraw it. ``rho = 1`` is full stickiness, ``rho = 0`` is the naive
independent model.

The susceptible share remaining after ``N`` identical rounds is

    s(N) = pi_zero + (1 - pi_zero) * [ rho * E_Beta[(1 - tau*r)^N]
                                     + (1 - rho) * (1 - tau*mu)^N ]

where ``tau`` is the per-dose take. Because ``(1 - tau*r)^N`` is convex in
``r``, Jensen's inequality makes the first term at least the second: any
stickiness slows the decay. The ``pi_zero`` term does not decay at all, so it
sets a hard immunity ceiling that no number of rounds can pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import roots_jacobi


# --------------------------------------------------------------------------
# Quadrature over the Beta reach distribution
# --------------------------------------------------------------------------


def beta_quadrature(a: float, b: float, n_nodes: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes and weights that integrate a function against ``Beta(a, b)``.

    Gauss-Jacobi quadrature is used because the Beta density is exactly the
    Jacobi weight function after mapping ``[-1, 1]`` to ``[0, 1]``. For a
    polynomial integrand of degree at most ``2*n_nodes - 1`` the result is
    exact rather than approximate, and ``(1 - tau*r)^N`` is a polynomial of
    degree ``N``. With the default node count every round budget up to 47 is
    integrated exactly.

    Args:
        a: First Beta shape parameter. Must be positive.
        b: Second Beta shape parameter. Must be positive.
        n_nodes: Number of quadrature nodes.

    Returns:
        A pair ``(nodes, weights)``. Nodes lie in ``(0, 1)``, weights sum to 1.
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"Beta shapes must be positive, got a={a}, b={b}.")
    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2.")

    # scipy's Jacobi weight on [-1, 1] is (1 - x)^alpha (1 + x)^beta.
    # Mapping r = (x + 1) / 2 turns it into r^(a-1) (1 - r)^(b-1) up to a
    # constant, so alpha = b - 1 and beta = a - 1.
    x, w = roots_jacobi(n_nodes, b - 1.0, a - 1.0)
    nodes = 0.5 * (x + 1.0)
    weights = w / w.sum()
    return nodes, weights


def beta_shapes(mu: float, kappa: float) -> tuple[float, float]:
    """Convert a mean and a concentration into Beta shape parameters.

    ``kappa`` is ``a + b``. A large ``kappa`` concentrates reach around ``mu``,
    meaning teams treat every child alike. A small ``kappa`` spreads it, meaning
    some children are reached almost always and others almost never.
    """
    mu = float(np.clip(mu, 1e-6, 1.0 - 1e-6))
    kappa = max(float(kappa), 2.0 + 1e-6)
    return mu * kappa, (1.0 - mu) * kappa


@dataclass(frozen=True)
class ReachDistribution:
    """Per-round reach across the children of one geographic unit.

    Attributes:
        mu: Mean per-round reach among children outside the unreachable core.
        kappa: Beta concentration. Lower means more unequal reach.
        pi_zero: Share of children never reached in any round.
        rho: Share of children whose reach propensity persists between rounds.
    """

    mu: float
    kappa: float
    pi_zero: float
    rho: float

    def __post_init__(self) -> None:
        for name, value in (
            ("mu", self.mu),
            ("pi_zero", self.pi_zero),
            ("rho", self.rho),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}.")
        if self.kappa <= 2.0:
            raise ValueError(f"kappa must exceed 2, got {self.kappa}.")

    @property
    def mean_population_reach(self) -> float:
        """Mean per-round reach across all children, unreachable core included."""
        return (1.0 - self.pi_zero) * self.mu

    def nodes_and_weights(self, n_nodes: int = 24) -> tuple[np.ndarray, np.ndarray]:
        """Discretise the full reach distribution, unreachable core included.

        The returned arrays describe a discrete distribution over reach
        propensity with an explicit atom at zero. Downstream code carries a
        susceptible mass at each node, so births and campaign pulses act on
        the right children.
        """
        a, b = beta_shapes(self.mu, self.kappa)
        nodes, weights = beta_quadrature(a, b, n_nodes)
        all_nodes = np.concatenate([[0.0], nodes])
        all_weights = np.concatenate([[self.pi_zero], (1.0 - self.pi_zero) * weights])
        return all_nodes, all_weights

    def susceptible_share_after(self, n_rounds: int, take: float, n_nodes: int = 24) -> float:
        """Share of an initially susceptible cohort still susceptible after ``n_rounds``.

        Assumes the cohort is closed: no births, no ageing out. Campaign rounds
        are identical and evenly spaced. Use ``immunity.project_trajectory`` when
        the cohort is open, which it always is in practice.
        """
        if n_rounds < 0:
            raise ValueError("n_rounds must not be negative.")
        if not 0.0 < take <= 1.0:
            raise ValueError("take must lie in (0, 1].")
        if n_rounds == 0:
            return 1.0

        a, b = beta_shapes(self.mu, self.kappa)
        nodes, weights = beta_quadrature(a, b, n_nodes)
        sticky = float(np.sum(weights * (1.0 - take * nodes) ** n_rounds))
        independent = (1.0 - take * self.mu) ** n_rounds
        reachable = self.rho * sticky + (1.0 - self.rho) * independent
        return self.pi_zero + (1.0 - self.pi_zero) * reachable

    def immunity_ceiling(self, initial_susceptible_share: float) -> float:
        """Highest population immunity that campaigns alone can ever reach.

        As the round count grows without bound the reachable susceptibles vanish
        and the unreachable core remains. This is the quantity that decides
        whether the answer to "how many campaigns" is a number or a different
        question.

        Args:
            initial_susceptible_share: Susceptible share of the population now.

        Returns:
            The asymptotic population immunity, in ``[0, 1]``.
        """
        if not 0.0 <= initial_susceptible_share <= 1.0:
            raise ValueError("initial_susceptible_share must lie in [0, 1].")
        return 1.0 - initial_susceptible_share * self.pi_zero


# --------------------------------------------------------------------------
# Estimating stickiness from chronic-miss history
# --------------------------------------------------------------------------


def estimate_persistence(
    misses: np.ndarray,
    rounds_targeted: np.ndarray,
    *,
    min_units: int = 20,
) -> tuple[float, str]:
    """Estimate ``rho`` from how repeatedly the same settlements are missed.

    If a settlement being missed in one round told us nothing about the next,
    the count of missed rounds would be Binomial and its variance would be
    ``K*p*(1-p)``. Observed counts are over-dispersed relative to that, and the
    intraclass correlation implied by the excess variance is exactly the
    stickiness the projection needs.

        Var(k) = K*p*(1-p) * [1 + (K-1)*rho]

    This is the method-of-moments estimator for the beta-binomial intraclass
    correlation, applied to the chronic-miss panel that the NEOC portal already
    publishes as ``core.v_chronic_miss``.

    Args:
        misses: Per settlement, the number of rounds classified sub-threshold.
        rounds_targeted: Per settlement, the number of rounds it was targeted in.
        min_units: Fewest settlements the estimator will accept.

    Returns:
        A pair ``(rho, note)``. ``note`` records how the value was obtained,
        including any fallback, so the report can carry it.
    """
    misses = np.asarray(misses, dtype=float)
    rounds_targeted = np.asarray(rounds_targeted, dtype=float)
    if misses.shape != rounds_targeted.shape:
        raise ValueError("misses and rounds_targeted must have the same shape.")

    keep = rounds_targeted >= 2
    misses, rounds_targeted = misses[keep], rounds_targeted[keep]

    if len(misses) < min_units:
        return 0.60, (
            f"rho set to the programme prior 0.60: only {len(misses)} settlements had "
            f"two or more targeted rounds, below the minimum of {min_units}."
        )

    # A common round count keeps the moment estimator unbiased. Take the mode,
    # which is the round budget most settlements actually experienced.
    values, counts = np.unique(rounds_targeted, return_counts=True)
    k_common = float(values[np.argmax(counts)])
    sel = rounds_targeted == k_common
    k_obs = misses[sel]

    if len(k_obs) < min_units or k_common < 2:
        return 0.60, (
            "rho set to the programme prior 0.60: no single round count had enough "
            "settlements for a stable moment estimate."
        )

    p_hat = float(k_obs.mean() / k_common)
    if p_hat <= 0.0 or p_hat >= 1.0:
        return 0.60, (
            f"rho set to the programme prior 0.60: every settlement had the same outcome "
            f"(p_hat = {p_hat:.2f}), so no dispersion is observable."
        )

    var_obs = float(k_obs.var(ddof=1))
    var_binomial = k_common * p_hat * (1.0 - p_hat)
    if var_binomial <= 0.0:
        return 0.60, "rho set to the programme prior 0.60: binomial variance was zero."

    rho = (var_obs / var_binomial - 1.0) / (k_common - 1.0)
    rho_clipped = float(np.clip(rho, 0.0, 0.98))
    note = (
        f"rho = {rho_clipped:.2f} estimated from {len(k_obs)} settlements over "
        f"{int(k_common)} rounds. Observed variance {var_obs:.3f} against binomial "
        f"{var_binomial:.3f}, a dispersion ratio of {var_obs / var_binomial:.2f}."
    )
    if rho < 0.0:
        note += " Negative raw estimate clipped to 0: misses were under-dispersed."
    if rho > 0.98:
        note += " Raw estimate clipped to 0.98: near-total stickiness."
    return rho_clipped, note


def estimate_concentration(
    reach_by_unit: np.ndarray,
    *,
    min_units: int = 10,
    verification_lot_size: int | None = None,
) -> tuple[float, str]:
    """Estimate the Beta concentration from the spread of reach across units.

    Method of moments: for ``Beta(a, b)`` with mean ``mu`` and variance ``v``, the
    concentration is ``mu*(1-mu)/v - 1``.

    The correction that matters is subtracting measurement noise. Observed reach
    comes from a small verification lot, so its spread across settlements is the
    real spread *plus* the sampling error of each lot. Feeding the raw spread into
    the formula therefore reports far more inequality between children than
    exists, which makes the model give up on hard-to-reach children too early and
    understates what campaigns can do. With a lot of 60 the sampling variance is
    of the same order as the real between-settlement variance, so the correction
    is not a refinement; it is the difference between a usable estimate and a
    wrong one.

    Args:
        reach_by_unit: Observed per-round reach for each sub-unit, in ``[0, 1]``.
        min_units: Fewest sub-units the estimator will accept.
        verification_lot_size: Sample size behind each observation. When given,
            its binomial sampling variance is removed from the observed spread.

    Returns:
        A pair ``(kappa, note)``.
    """
    x = np.asarray(reach_by_unit, dtype=float)
    x = x[np.isfinite(x)]
    x = np.clip(x, 1e-4, 1.0 - 1e-4)

    if len(x) < min_units:
        return 8.0, (
            f"kappa set to the programme prior 8.0: only {len(x)} sub-units had usable "
            f"reach, below the minimum of {min_units}."
        )

    mu, var_observed = float(x.mean()), float(x.var(ddof=1))
    if var_observed <= 0.0:
        return 50.0, "kappa set to 50 (near-uniform reach): observed variance was zero."

    sampling_variance = 0.0
    if verification_lot_size and verification_lot_size > 1:
        sampling_variance = mu * (1.0 - mu) / verification_lot_size

    var_true = var_observed - sampling_variance
    if var_true <= 1e-6:
        return 50.0, (
            f"kappa set to 50: the spread across {len(x)} sub-units "
            f"({np.sqrt(var_observed):.3f}) is no larger than the sampling noise of a "
            f"lot of {verification_lot_size}, so no real inequality is detectable."
        )

    kappa = mu * (1.0 - mu) / var_true - 1.0
    kappa_clipped = float(np.clip(kappa, 2.5, 200.0))
    detail = (
        f" after removing sampling variance {sampling_variance:.4f} of an observed "
        f"{var_observed:.4f}"
        if sampling_variance
        else " (no lot size supplied, so sampling noise is still in the estimate)"
    )
    return kappa_clipped, (
        f"kappa = {kappa_clipped:.1f} from {len(x)} sub-units, mean reach {mu:.3f}{detail}."
    )


def estimate_persistence_from_reach(
    unit_ids: np.ndarray,
    round_ids: np.ndarray,
    reach: np.ndarray,
    *,
    verification_lot_size: int | None = None,
    campaign_wobble_cv: float = 0.0,
    min_units: int = 20,
) -> tuple[float, str]:
    """Estimate ``rho`` from how much of the variation in reach is between places.

    This is the preferred estimator. The chronic-miss version below works on a
    pass or fail label, and thresholding a continuous quantity throws away most of
    the signal about persistence: two settlements can both fail every round while
    one is at 79% and the other at 30%. Measured on the continuous reach series
    instead, persistence is exactly the intraclass correlation, which is the share
    of variation that sits between settlements rather than within one over time.

    Round effects are removed first, so a nationally bad round does not read as
    every settlement being persistently bad. Verification sampling noise is
    removed from the within-settlement term where the lot size is known, because
    otherwise measurement error looks like a settlement changing between rounds
    and drags the estimate down.

    Args:
        unit_ids: Unit identifier for each observation.
        round_ids: Round identifier for each observation.
        reach: Observed reach for each observation.
        verification_lot_size: Sample size behind each observation.
        campaign_wobble_cv: Round-to-round movement in a settlement's own reach,
            from ``estimate_reach_wobble``. Removed from the within-settlement term
            for the same reason sampling noise is: a settlement whose reach moved
            because a team changed is not a settlement whose children reshuffled.
            Leaving it in can only push the estimate down, so this correction has
            a known direction.
        min_units: Fewest units the estimator will accept.

    Returns:
        A pair ``(rho, note)``.
    """
    unit_ids = np.asarray(unit_ids)
    round_ids = np.asarray(round_ids)
    reach = np.asarray(reach, dtype=float)

    keep = np.isfinite(reach)
    unit_ids, round_ids, reach = unit_ids[keep], round_ids[keep], reach[keep]

    units, unit_index = np.unique(unit_ids, return_inverse=True)
    if len(units) < min_units:
        return 0.60, (
            f"rho set to the programme prior 0.60: only {len(units)} units had usable "
            f"reach, below the minimum of {min_units}."
        )

    # Remove the round effect so a bad national round is not read as persistence.
    rounds, round_index = np.unique(round_ids, return_inverse=True)
    round_mean = np.bincount(round_index, weights=reach) / np.bincount(round_index)
    centred = reach - round_mean[round_index] + reach.mean()

    counts = np.bincount(unit_index)
    unit_mean = np.bincount(unit_index, weights=centred) / counts
    grand_mean = centred.mean()

    usable = counts >= 2
    if usable.sum() < min_units:
        return 0.60, (
            f"rho set to the programme prior 0.60: only {int(usable.sum())} units were "
            "observed in two or more rounds."
        )

    # One-way random-effects decomposition.
    between = float(np.sum(counts * (unit_mean - grand_mean) ** 2))
    within = float(np.sum((centred - unit_mean[unit_index]) ** 2))
    df_between = len(units) - 1
    df_within = len(centred) - len(units)
    if df_between < 1 or df_within < 1:
        return 0.60, "rho set to the programme prior 0.60: not enough degrees of freedom."

    ms_between = between / df_between
    ms_within = within / df_within

    if verification_lot_size and verification_lot_size > 1:
        sampling_variance = grand_mean * (1.0 - grand_mean) / verification_lot_size
        ms_within = max(ms_within - sampling_variance, 1e-8)

    # Stickiness is a statement about which children a round misses. It has no
    # effect at all on how many a settlement reaches, so movement in the
    # settlement's own aggregate reach is not evidence about it. Left in the
    # within term it reads as children reshuffling and drags the estimate down.
    wobble_variance = 0.0
    if campaign_wobble_cv > 0.0:
        wobble_variance = (campaign_wobble_cv * grand_mean) ** 2
        ms_within = max(ms_within - wobble_variance, 1e-8)

    # Effective group size for unbalanced designs.
    n_effective = (len(centred) - float(np.sum(counts**2)) / len(centred)) / df_between
    n_effective = max(n_effective, 1.0 + 1e-6)

    variance_between = max((ms_between - ms_within) / n_effective, 0.0)
    total = variance_between + ms_within
    if total <= 0:
        return 0.60, "rho set to the programme prior 0.60: total variance was zero."

    rho = float(np.clip(variance_between / total, 0.0, 0.98))
    correction = (
        f", after removing the sampling variance of a lot of {verification_lot_size}"
        if verification_lot_size
        else ", with verification sampling noise still counted as within-unit variation, "
             "which biases this estimate downward"
    )
    if wobble_variance > 0.0:
        correction += (
            f" and the {campaign_wobble_cv:.1%} round-to-round movement in the settlement's "
            "own reach, which stickiness does not cause"
        )
    return rho, (
        f"rho = {rho:.2f} from the intraclass correlation of reach across "
        f"{len(units)} units and {len(rounds)} rounds{correction}."
    )


def estimate_reach_wobble(
    unit_ids: np.ndarray,
    round_ids: np.ndarray,
    admin_reach: np.ndarray,
    verified_reach: np.ndarray,
    *,
    cap: float = 0.40,
    min_units: int = 20,
    min_rounds_per_unit: int = 4,
) -> tuple[float, str]:
    """Estimate how much a settlement's own reach moves from one round to the next.

    The engine needs this and does not otherwise have it. Stickiness decides which
    children a round misses, never how many, so the simulator holds a settlement's
    aggregate reach fixed across every round of a draw. A draw then either clears
    the target in the first few rounds or never clears it, and the round count has
    no middle. Real campaigns are not like that: a team changes, a road closes, a
    market day falls badly, and the same settlement returns a different number.

    Measuring it is the hard part, because the obvious estimator is wrong. The
    round-to-round spread of a single reported series is mostly reporting noise,
    and feeding that spread to the simulator makes the engine worse, not better:
    an inflated shock lets an unlucky draw clear the target on one good round, and
    a first crossing is recorded whether or not it would have held.

    The panel carries two streams of the same underlying round, and their errors
    are independent - one is what teams tallied, the other is what verification
    found. Their covariance within a settlement therefore keeps what the two
    streams agree on, which is the round itself, and drops what only one of them
    saw, which is noise. In logs, that covariance is the squared coefficient of
    variation directly.

    The median across settlements is taken rather than the mean. Accessibility
    moves between rounds, and a settlement that closes and reopens produces a
    covariance orders of magnitude above the rest. That movement is real, but it
    belongs to the unreachable core and to the reach model, not to the ordinary
    round-to-round wobble this parameter carries, and a mean would let a handful
    of such settlements set the figure for every unit in the programme.

    Args:
        unit_ids: Unit identifier for each observation.
        round_ids: Round identifier for each observation.
        admin_reach: Administrative coverage for each observation.
        verified_reach: Verified coverage for each observation.
        cap: Upper bound on the returned coefficient of variation.
        min_units: Fewest units with enough rounds that the estimator will accept.
        min_rounds_per_unit: Rounds a unit needs before it contributes. Four is the
            floor at which a within-unit covariance means anything.

    Returns:
        A pair ``(coefficient_of_variation, note)``.
    """
    unit_ids = np.asarray(unit_ids)
    round_ids = np.asarray(round_ids)
    admin = np.asarray(admin_reach, dtype=float)
    verified = np.asarray(verified_reach, dtype=float)

    fallback = (
        0.0,
        "Round-to-round reach movement not estimated, so reach is held steady between "
        "rounds. The round count will have a thinner middle than the field does.",
    )

    keep = np.isfinite(admin) & np.isfinite(verified) & (admin > 0.01) & (verified > 0.01)
    if keep.sum() < min_units * min_rounds_per_unit:
        return fallback
    unit_ids, round_ids = unit_ids[keep], round_ids[keep]
    log_admin, log_verified = np.log(admin[keep]), np.log(verified[keep])

    # A nationally good or bad round is a shared shock, not a settlement's own
    # movement, so it is removed from each stream before anything else.
    _, round_index = np.unique(round_ids, return_inverse=True)
    for series in (log_admin, log_verified):
        round_mean = np.bincount(round_index, weights=series) / np.bincount(round_index)
        series -= round_mean[round_index]

    units, unit_index = np.unique(unit_ids, return_inverse=True)
    counts = np.bincount(unit_index)
    unit_admin = np.bincount(unit_index, weights=log_admin) / counts
    unit_verified = np.bincount(unit_index, weights=log_verified) / counts
    product = (log_admin - unit_admin[unit_index]) * (log_verified - unit_verified[unit_index])

    covariance = np.bincount(unit_index, weights=product) / np.maximum(counts - 1, 1)
    usable = counts >= min_rounds_per_unit
    if usable.sum() < min_units:
        return fallback

    shared = float(np.median(covariance[usable]))
    if shared <= 0.0:
        return (
            0.0,
            f"The two reporting streams share no round-to-round movement across "
            f"{int(usable.sum())} settlements, so reach is held steady between rounds. "
            "Either campaigns are unusually uniform here or one stream is not measuring "
            "the round at all.",
        )

    cv = float(np.clip(np.sqrt(shared), 0.0, cap))
    capped = " (capped)" if cv >= cap else ""
    return cv, (
        f"Reach moves {cv:.1%} of its own level between rounds within a settlement{capped}, "
        f"from the movement the administrative and verified streams agree on across "
        f"{int(usable.sum())} settlements. Taken as the median so that settlements which "
        "close and reopen do not set the figure for everywhere else."
    )


def separate_persistent_reach_spread(
    draws: np.ndarray,
    wobble_cv: float,
    *,
    variance_floor: float = 0.10,
) -> tuple[np.ndarray, str]:
    """Take the round-to-round movement back out of the predictive reach spread.

    The reach model is trained on the reach of a single round, so what it
    predicts, and what its interval covers, is what the next round will return.
    That spread holds two different things at once: how little is known about the
    settlement's own level, which persists for as long as the plan runs, and how
    much any one round moves around that level, which does not.

    The inversion needs them apart. It draws one reach per replicate, holds it
    across every round of that draw because the settlement's level is a property
    of the place, and then applies the round-to-round shock separately. Handing it
    the single-round spread therefore counts the movement twice: once inside the
    level that never changes, and again in the shock. The round-count
    distribution comes out too wide, and a distribution that is too wide is not
    merely cautious. It shrinks every stated probability toward the middle, so a
    settlement that will certainly not get there is given a chance it does not
    have, and one that certainly will is denied the confidence it has earned.

    The variances add, because a round's movement is independent of what is not
    known about the level, so the level's own variance is the difference between
    them. Draws are scaled about their median by the square root of the share
    that remains, which narrows the spread while keeping the centre and the
    skew that a bounded quantity has.

    The subtraction is floored rather than allowed to vanish. The wobble is one
    figure for the whole programme, while the predictive spread is conditional
    and can be narrow wherever the model is confident, so for some units the
    subtraction would take everything. Nothing about a campaign is known that
    exactly, and treating a level as certain would hand back the opposite error.

    Args:
        draws: Predictive reach draws, shape ``(n_units, n_draws)``.
        wobble_cv: Round-to-round movement in a settlement's reach, as a share of
            its level. Zero leaves the draws untouched.
        variance_floor: Least share of the predictive variance kept as the
            settlement's own level.

    Returns:
        A pair ``(draws, note)``.
    """
    draws = np.atleast_2d(np.asarray(draws, dtype=float))
    if wobble_cv <= 0.0:
        return draws, (
            "Reach draws carry the single-round spread unchanged, because no "
            "round-to-round movement was estimated to take out of it."
        )
    if not 0.0 < variance_floor <= 1.0:
        raise ValueError(f"variance_floor must lie in (0, 1], got {variance_floor}.")

    centre = np.median(draws, axis=1, keepdims=True)
    predictive_variance = draws.var(axis=1, keepdims=True)
    wobble_variance = (wobble_cv * centre) ** 2

    share = np.ones_like(predictive_variance)
    measurable = predictive_variance > 1e-12
    share[measurable] = 1.0 - wobble_variance[measurable] / predictive_variance[measurable]
    floored = share < variance_floor
    share = np.maximum(share, variance_floor)

    narrowed = centre + (draws - centre) * np.sqrt(share)
    at_floor = float(floored.mean())
    return np.clip(narrowed, 0.0, 1.0), (
        f"Round-to-round movement of {wobble_cv:.1%} taken back out of the reach "
        f"draws, which the model states for a single round and the inversion holds "
        f"across every round of a draw. The settlement's own spread narrows to "
        f"{float(np.median(np.sqrt(share))):.0%} of the single-round spread at the "
        f"median unit; {at_floor:.0%} of units sit at the floor, where the movement "
        "would otherwise account for the whole of it."
    )


def unreachable_core(
    *,
    inaccessible_share: float,
    refusal_share: float,
    nomadic_share: float,
    nomadic_miss_rate: float = 0.45,
) -> float:
    """Combine the three routes into the unreachable core into one share.

    The three are treated as overlapping rather than additive. A settlement that
    is both insecure and nomadic is one unreachable population, not two, so the
    shares are combined through their complements.

    Args:
        inaccessible_share: Share of the target population in settlements flagged
            Inaccessible.
        refusal_share: Share refusing in every round.
        nomadic_share: Share of the target population that is mobile.
        nomadic_miss_rate: Share of the mobile population that campaign
            microplans never enumerate.

    Returns:
        The share of children with zero reach in every round.
    """
    for name, value in (
        ("inaccessible_share", inaccessible_share),
        ("refusal_share", refusal_share),
        ("nomadic_share", nomadic_share),
        ("nomadic_miss_rate", nomadic_miss_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1], got {value}.")

    reachable = (
        (1.0 - inaccessible_share)
        * (1.0 - refusal_share)
        * (1.0 - nomadic_share * nomadic_miss_rate)
    )
    return float(np.clip(1.0 - reachable, 0.0, 0.999))
