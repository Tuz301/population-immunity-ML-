"""Tests of the epidemiology, not of the plumbing.

Each test names the modelling claim it protects. A test that only checks a
function returns a float protects nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from immunity_engine.contracts import EngineConfig
from immunity_engine.heterogeneity import (
    ReachDistribution,
    beta_quadrature,
    beta_shapes,
    estimate_concentration,
    estimate_persistence_from_reach,
    estimate_reach_wobble,
    separate_persistent_reach_spread,
    unreachable_core,
)
from immunity_engine.immunity import (
    apply_round,
    evolve,
    initial_state,
    project_trajectory,
    steady_state_immunity,
)
from immunity_engine.adapters.synthetic import SyntheticProgramme, generate_synthetic_panel
from immunity_engine.features import build_features
from immunity_engine.reach_model import ReachModel
from immunity_engine.vectorised import build_reach_grid, simulate_rounds


# --------------------------------------------------------------------------
# Quadrature
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_rounds", [1, 3, 7, 12])
@pytest.mark.parametrize(("mu", "kappa"), [(0.3, 4.0), (0.7, 8.0), (0.9, 30.0)])
def test_quadrature_is_exact_against_the_moment_expansion(n_rounds, mu, kappa):
    """Gauss-Jacobi integrates a polynomial in reach exactly, not approximately.

    ``(1 - tau*r)^N`` is a degree-N polynomial, so the quadrature must agree with
    the binomial expansion over the Beta moments to machine precision. If this
    ever fails, the susceptible-decay curve is quietly wrong everywhere.
    """
    a, b = beta_shapes(mu, kappa)
    nodes, weights = beta_quadrature(a, b, 24)
    tau = 0.5

    quadrature = float(np.sum(weights * (1.0 - tau * nodes) ** n_rounds))
    expansion = sum(
        math.comb(n_rounds, k)
        * (-tau) ** k
        * math.prod((a + j) / (a + b + j) for j in range(k))
        for k in range(n_rounds + 1)
    )
    assert quadrature == pytest.approx(expansion, abs=1e-12)


def test_batched_grid_matches_the_scalar_quadrature():
    """The fast path and the readable path must model the same distribution.

    The vectorised simulator discretises reach onto a shared cell grid so every
    draw can advance at once. That is only legitimate if it integrates to the
    same answer as the exact scalar quadrature.
    """
    reach = ReachDistribution(mu=0.72, kappa=9.0, pi_zero=0.05, rho=1.0)
    nodes, weights = build_reach_grid(
        np.array([0.72]), np.array([9.0]), np.array([0.05]), n_cells=48
    )
    take = 0.5
    for n_rounds in (1, 4, 10):
        batched = float(np.sum(weights[0] * (1.0 - take * nodes[0]) ** n_rounds))
        scalar = reach.susceptible_share_after(n_rounds, take)
        assert batched == pytest.approx(scalar, abs=2e-3)


# --------------------------------------------------------------------------
# Heterogeneity
# --------------------------------------------------------------------------


def test_stickiness_slows_the_decay_of_susceptibles():
    """Persistent reach must never look better than independent reach.

    Jensen's inequality guarantees it, because the survival factor is convex in
    reach. This is the mechanism behind diminishing returns, so a sign error here
    would make the engine recommend too few rounds everywhere.
    """
    sticky = ReachDistribution(mu=0.7, kappa=6.0, pi_zero=0.0, rho=1.0)
    independent = ReachDistribution(mu=0.7, kappa=6.0, pi_zero=0.0, rho=0.0)
    for n_rounds in (2, 5, 10):
        assert sticky.susceptible_share_after(n_rounds, 0.5) > independent.susceptible_share_after(
            n_rounds, 0.5
        )


def test_the_unreachable_core_sets_a_ceiling_no_round_count_passes():
    """A share of children reachable in no round caps immunity, permanently.

    This is the finding the engine exists to surface. If unlimited rounds ever
    drove susceptibles to zero here, the engine would recommend campaigns for
    places that need access negotiation instead.
    """
    reach = ReachDistribution(mu=0.85, kappa=10.0, pi_zero=0.10, rho=0.8)
    share = reach.susceptible_share_after(400, 0.6)
    assert share == pytest.approx(0.10, abs=1e-6)
    assert reach.immunity_ceiling(initial_susceptible_share=0.5) == pytest.approx(0.95)


def test_overlapping_barriers_are_not_added_twice():
    """A settlement that is both insecure and nomadic is one problem, not two."""
    core = unreachable_core(
        inaccessible_share=0.10, refusal_share=0.10, nomadic_share=0.20, nomadic_miss_rate=0.5
    )
    assert core < 0.10 + 0.10 + 0.10
    assert core == pytest.approx(1.0 - 0.9 * 0.9 * 0.9, abs=1e-9)


def test_concentration_estimator_removes_verification_sampling_noise():
    """Spread caused by a small verification lot is not inequality between children.

    Without the correction the estimator reports far more inequality than exists,
    which makes the engine give up on hard-to-reach children too early.
    """
    rng = np.random.default_rng(0)
    true_reach = np.clip(rng.beta(0.7 * 20, 0.3 * 20, 400), 1e-3, 1 - 1e-3)
    lot = 60
    observed = rng.binomial(lot, true_reach) / lot

    uncorrected, _ = estimate_concentration(observed)
    corrected, _ = estimate_concentration(observed, verification_lot_size=lot)
    assert corrected > uncorrected
    assert corrected == pytest.approx(20.0, rel=0.45)


def test_stickiness_estimator_recovers_a_known_value():
    """The intraclass correlation of reach is the stickiness the projection needs."""
    rng = np.random.default_rng(1)
    n_units, n_rounds, rho_true = 400, 8, 0.7
    between = math.sqrt(rho_true) * 0.10
    within = math.sqrt(1 - rho_true) * 0.10

    unit_level = rng.normal(0.65, between, n_units)
    unit_ids, round_ids, reach = [], [], []
    for u in range(n_units):
        for r in range(n_rounds):
            unit_ids.append(f"U{u}")
            round_ids.append(r)
            reach.append(np.clip(unit_level[u] + rng.normal(0, within), 0.01, 0.99))

    rho, _ = estimate_persistence_from_reach(
        np.array(unit_ids), np.array(round_ids), np.array(reach)
    )
    assert rho == pytest.approx(rho_true, abs=0.10)


# --------------------------------------------------------------------------
# Stock dynamics
# --------------------------------------------------------------------------


def test_without_campaigns_immunity_settles_at_routine_immunisation_coverage():
    """The no-campaign equilibrium is a property of the health system, not a guess.

    With births flowing in and children ageing out, population immunity settles at
    the share of newborns routine immunisation protects. The whole engine uses
    that level as the baseline it replays history from, so it has to be right.
    """
    reach = ReachDistribution(mu=0.7, kappa=8.0, pi_zero=0.02, rho=0.7)
    state = initial_state(total_children=10000.0, population_immunity=0.0, reach=reach)
    ri = 0.55
    for _ in range(200):
        state = evolve(
            state, months=6.0, births_per_month=10000 / 60, ri_protection=ri, age_band_months=60
        )
    assert state.population_immunity == pytest.approx(ri, abs=0.01)


def test_a_round_never_reaches_the_unreachable_core():
    """The atom at zero reach must survive every pulse untouched."""
    reach = ReachDistribution(mu=0.8, kappa=10.0, pi_zero=0.08, rho=1.0)
    state = initial_state(total_children=1000.0, population_immunity=0.0, reach=reach)
    before = state.susceptible[0]
    after = apply_round(state, take=0.9, reach=reach)
    assert after.susceptible[0] == pytest.approx(before)


def test_shorter_intervals_hold_more_immunity_between_rounds():
    """Births refill the stock, so the trough depends on the interval, not the count."""
    reach = ReachDistribution(mu=0.8, kappa=8.0, pi_zero=0.02, rho=0.7)
    shared = dict(
        take=0.5,
        reach=reach,
        births_per_month=10000 / 60,
        ri_protection=0.5,
        total_children=10000.0,
    )
    _, trough_fast = steady_state_immunity(interval_months=3.0, **shared)
    _, trough_slow = steady_state_immunity(interval_months=12.0, **shared)
    assert trough_fast > trough_slow


def test_immunity_rises_monotonically_under_a_repeating_schedule():
    """Each round removes susceptibles; nothing in the model can put them back."""
    reach = ReachDistribution(mu=0.75, kappa=8.0, pi_zero=0.03, rho=0.7)
    state = initial_state(total_children=20000.0, population_immunity=0.30, reach=reach)
    trajectory = project_trajectory(
        state,
        n_rounds=10,
        take=0.5,
        reach=reach,
        interval_months=3.0,
        births_per_month=20000 / 60,
        ri_protection=0.5,
    )
    assert np.all(np.diff(trajectory.immunity_after_round) > -1e-9)


# --------------------------------------------------------------------------
# Simulator behaviour
# --------------------------------------------------------------------------


def test_falling_reach_never_shortens_the_plan():
    """Projected fatigue must not make a place look easier than a stationary one."""
    shared = dict(
        mu=np.array([0.75]),
        kappa=np.array([8.0]),
        pi_zero=np.array([0.03]),
        rho=np.array([0.7]),
        take=np.array([0.5]),
        initial_immunity=np.array([0.5]),
        cohort=np.array([20000.0]),
        births_per_month=np.array([20000 / 60]),
        ri_protection=0.5,
        interval_months=3.0,
        target_immunity=0.90,
        max_rounds=20,
    )
    stationary, _ = simulate_rounds(reach_drift_per_round=0.0, **shared)
    declining, _ = simulate_rounds(reach_drift_per_round=-0.03, **shared)
    assert declining[0] >= stationary[0]


def test_a_higher_target_never_needs_fewer_rounds():
    """A monotonicity the answer must satisfy whatever the parameters are."""
    shared = dict(
        mu=np.array([0.8]),
        kappa=np.array([8.0]),
        pi_zero=np.array([0.02]),
        rho=np.array([0.7]),
        take=np.array([0.5]),
        initial_immunity=np.array([0.4]),
        cohort=np.array([10000.0]),
        births_per_month=np.array([10000 / 60]),
        ri_protection=0.5,
        interval_months=3.0,
        max_rounds=30,
    )
    low, _ = simulate_rounds(target_immunity=0.80, **shared)
    high, _ = simulate_rounds(target_immunity=0.92, **shared)
    assert high[0] >= low[0]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_the_programme_immunity_target_reproduces_the_published_figure():
    """Vc(adj) at R0 = 6 and epsilon = 0.126 is the programme's 95.3%."""
    assert EngineConfig().vc_adjusted == pytest.approx(0.953, abs=0.001)


def test_per_dose_take_and_schedule_failure_are_consistent():
    """The two vaccine parameters describe the same vaccine and must agree.

    A schedule failure of 0.126 over three doses implies a per-dose take of
    0.499. The default per-dose take is 0.500, so the residual is a tenth of a
    percentage point. If someone changes one without the other, this catches it.
    """
    reconciliation = EngineConfig().reconcile_take()
    assert reconciliation["per_dose_take_implied_by_epsilon"] == pytest.approx(0.4987, abs=1e-3)
    assert abs(reconciliation["residual_pp"]) < 0.5


def test_a_reproduction_number_below_one_is_refused():
    """Below R0 = 1 there is no herd-immunity threshold, so there is no target."""
    with pytest.raises(ValueError, match="r0 must exceed 1"):
        EngineConfig(r0=0.9)


def test_wobble_estimator_keeps_only_what_both_streams_agree_on():
    """Reporting noise is independent between streams; a real round moves both.

    The estimator must recover the shared movement and ignore the rest, because an
    inflated shock lets an unlucky draw clear the target on one good round.
    """
    rng = np.random.default_rng(7)
    n_units, n_rounds, true_cv = 200, 8, 0.09
    unit_ids, round_ids, admin, verified = [], [], [], []
    for u in range(n_units):
        base = rng.uniform(0.35, 0.85)
        for r in range(1, n_rounds + 1):
            shared = base * rng.lognormal(-0.5 * true_cv**2, true_cv)
            unit_ids.append(f"U{u}")
            round_ids.append(r)
            # Independent reporting error, three times the size of the real signal.
            admin.append(shared * rng.lognormal(0.0, 0.27))
            verified.append(shared * rng.lognormal(0.0, 0.27))

    cv, note = estimate_reach_wobble(
        np.array(unit_ids), np.array(round_ids), np.array(admin), np.array(verified)
    )
    assert abs(cv - true_cv) < 0.03, f"recovered {cv:.3f} against a true {true_cv}: {note}"

    # The same movement measured from one stream alone is swamped by its own noise.
    single, _ = estimate_reach_wobble(
        np.array(unit_ids), np.array(round_ids), np.array(admin), np.array(admin)
    )
    assert single > cv * 2, "one stream cannot separate campaign movement from noise"


def test_stickiness_is_not_dragged_down_by_movement_it_does_not_cause():
    """Stickiness decides which children a round misses, never how many.

    Movement in a settlement's own aggregate reach is therefore not evidence about
    it, and leaving that movement in the within-unit term can only push the
    estimate down. The correction has a known direction, so the test asserts it.
    """
    rng = np.random.default_rng(11)
    n_units, n_rounds = 300, 8
    unit_ids, round_ids, reach = [], [], []
    for u in range(n_units):
        base = rng.uniform(0.30, 0.90)
        for r in range(1, n_rounds + 1):
            unit_ids.append(f"U{u}")
            round_ids.append(r)
            reach.append(np.clip(base * rng.lognormal(0.0, 0.12), 0.01, 0.99))

    units, rounds_, values = np.array(unit_ids), np.array(round_ids), np.array(reach)
    uncorrected, _ = estimate_persistence_from_reach(units, rounds_, values)
    corrected, note = estimate_persistence_from_reach(
        units, rounds_, values, campaign_wobble_cv=0.12
    )
    assert corrected > uncorrected, note
    assert corrected <= 0.98


def test_interval_shape_is_read_off_rows_that_taught_neither_model():
    """The spread model must not be asked to score its own training residuals.

    Dividing a model's own training residuals by its own predictions understates
    them, and every interval built from that runs narrow. The failure hides,
    because coverage on the slice that produced it looks correct.
    """
    panel, _ = generate_synthetic_panel(SyntheticProgramme(settlements_per_ward=3, n_rounds=8))
    config = EngineConfig()
    features, _ = build_features(panel, config)

    model = ReachModel()
    report = model.fit(features, validation_rounds=2)

    for name, observed in report.interval_coverage.items():
        nominal = float(name.rstrip("%")) / 100.0
        assert abs(observed - nominal) < 0.12, (
            f"the {name} interval covered {observed:.0%} on rounds it had never seen"
        )


def test_round_movement_is_taken_back_out_of_the_persistent_reach_spread():
    """The model states a single round; the inversion holds a level.

    A settlement's predicted reach spread holds both how little is known about the
    place and how much any one round moves. The inversion draws the level once and
    applies the movement separately, so handing it the single-round spread counts
    the movement twice and widens every round count.
    """
    rng = np.random.default_rng(5)
    level_sd, wobble_cv, centre = 0.06, 0.08, 0.60
    wobble_sd = wobble_cv * centre
    single_round = centre + rng.normal(0.0, math.hypot(level_sd, wobble_sd), size=(1, 40000))

    narrowed, note = separate_persistent_reach_spread(single_round, wobble_cv)

    assert narrowed.std() < single_round.std(), note
    assert abs(narrowed.std() - level_sd) < 0.006, (
        f"level spread came back {narrowed.std():.4f} against a true {level_sd}: {note}"
    )
    # The centre is what the model forecast, and narrowing must not move it.
    assert abs(float(np.median(narrowed)) - float(np.median(single_round))) < 1e-9

    # With no movement estimated there is nothing to take out.
    untouched, _ = separate_persistent_reach_spread(single_round, 0.0)
    assert np.allclose(untouched, single_round)


def test_the_spread_of_a_level_is_never_taken_all_the_way_to_zero():
    """The movement is one figure for the programme; the spread is per unit.

    Where the two would cancel, the honest reading is that the level is well
    known, not that it is certain. A level asserted without error would hand back
    the same defect from the other side.
    """
    confident = np.full((1, 2000), 0.70) + np.random.default_rng(3).normal(0, 0.005, (1, 2000))
    narrowed, note = separate_persistent_reach_spread(confident, 0.25)
    assert narrowed.std() > 0.0, note
    assert "floor" in note


def test_the_uncertainty_on_current_immunity_answers_to_its_inputs():
    """A flat uncertainty cannot follow an error that is not flat.

    Immunity is rebuilt from the rounds already run, so what is not known about
    it comes from what is not known about those inputs. Near the ceiling a shift
    in reach barely moves it; far from the ceiling every input does. An estimator
    that returns the same figure for both is over-confident where the decision is
    hardest and over-cautious where it is easiest.
    """
    from immunity_engine.contracts import ProvenanceLog
    from immunity_engine.pipeline import estimate_current_immunity, estimate_unit_parameters

    panel, _ = generate_synthetic_panel(SyntheticProgramme(seed=19))
    config = EngineConfig()
    log = ProvenanceLog()
    features, log = build_features(panel, config, log)
    parameters, _ = estimate_unit_parameters(features, panel, config, log)
    frame = estimate_current_immunity(features, parameters, panel, config)

    spread = frame["current_immunity_sd"].to_numpy()
    assert np.all(np.isfinite(spread)) and np.all(spread > 0.0)
    assert spread.std() > 1e-3, "the uncertainty is still effectively one number for every unit"

    # Confidence must grow as immunity approaches the ceiling, where a round can
    # no longer move it and the reconstruction has less room to be wrong.
    level = frame["current_immunity"].to_numpy()
    usable = np.isfinite(level) & np.isfinite(spread)
    assert np.corrcoef(level[usable], spread[usable])[0, 1] < 0.0, (
        "the engine is not more certain about settlements sitting near the ceiling"
    )


def test_an_infeasible_verdict_never_prints_a_ceiling_that_reaches_the_target():
    """The verdict and the ceiling shown beside it must agree at the boundary.

    The verdict fires when at least half the draws put the ceiling below the
    target. Reporting the midpoint of the two middle draws breaks that agreement
    exactly at half, where the two straddle the target: the engine then says no
    number of rounds will reach it while printing a ceiling that does, and the
    sentence a programme manager reads contradicts itself.

    Constructed rather than simulated, because a run only lands on the boundary
    by luck and the invariant has to hold every time.
    """
    from immunity_engine.rounds_required import _lower_median

    target = 0.95
    # Exactly half the draws below the target, and the half above sitting further
    # from it than the half below. This is the case that inverts the report.
    ceilings = np.array([0.949, 0.949, 0.990, 0.990])
    assert float(np.mean(ceilings < target)) >= 0.50, "the verdict must fire here"
    assert np.median(ceilings) > target, "averaging the middle pair is what breaks it"
    assert _lower_median(ceilings) < target

    # The agreement must hold for any draw count and any arrangement.
    rng = np.random.default_rng(31)
    for _ in range(400):
        size = int(rng.integers(2, 400))
        sample = rng.uniform(0.80, 0.999, size=size)
        if float(np.mean(sample < target)) >= 0.50:
            assert _lower_median(sample) < target, (
                f"{size} draws put the ceiling below target at least half the time, "
                f"yet the reported ceiling was {_lower_median(sample):.4f}"
            )
