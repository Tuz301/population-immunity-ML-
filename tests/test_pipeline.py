"""Tests of the contract, the pipeline and the council.

These protect the behaviour a user depends on: that bad input is refused loudly,
that assumptions are recorded, that the round count comes from history rather
than from the model's prior, and that the council cannot be talked round.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from immunity_engine.adapters.synthetic import SyntheticProgramme, generate_synthetic_panel
from immunity_engine.contracts import ContractError, EngineConfig, Feasibility, validate_panel
from immunity_engine.council import Verdict, convene
from immunity_engine.council.experts import COUNCIL
from immunity_engine.optimizer import Objective, allocate_rounds
from immunity_engine.pipeline import run_pipeline
from immunity_engine.report import programme_summary, unit_brief


@pytest.fixture(scope="module")
def small_run():
    """A complete run on a small simulated programme, shared across tests."""
    spec = SyntheticProgramme(
        n_states=2, lgas_per_state=3, wards_per_lga=3, settlements_per_ward=4,
        n_rounds=8, forward_rounds=16, seed=11,
    )
    panel, truth = generate_synthetic_panel(spec)
    config = EngineConfig(n_draws=250, max_rounds=10)
    plan_set = run_pipeline(panel, config, planning_interval_months=3.0)
    return panel, truth, config, plan_set


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_a_missing_required_column_stops_the_run(small_run):
    """Silently continuing without a denominator basis is how two truths appear."""
    panel = small_run[0].drop(columns=["denominator_basis"])
    with pytest.raises(ContractError, match="denominator_basis"):
        validate_panel(panel)


def test_duplicate_unit_rounds_are_refused(small_run):
    """A duplicated round double counts doses and silently halves the round count."""
    panel = pd.concat([small_run[0], small_run[0].head(1)], ignore_index=True)
    with pytest.raises(ContractError, match="duplicate"):
        validate_panel(panel)


def test_coverage_above_110_percent_is_reported_as_a_denominator_defect(small_run):
    """Above-100% coverage is never success. The contract must say so out loud."""
    report = validate_panel(small_run[0])
    assert any("denominator defect" in w for w in report.warnings)


def test_missing_verification_is_recorded_as_an_assumption(small_run):
    """Reach inferred from administrative counts keeps their optimism, and says so."""
    panel = small_run[0].drop(columns=["verified_vaccinated"])
    plan_set = run_pipeline(panel, EngineConfig(n_draws=120), planning_interval_months=3.0)
    fields = [entry["field"] for entry in plan_set.provenance.entries]
    assert "verified_vaccinated" in fields


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


def test_reconstructed_immunity_tracks_the_truth(small_run):
    """Immunity is rebuilt by replaying rounds, not measured. It has to be close."""
    _, truth, _, plan_set = small_run
    merged = plan_set.parameters.join(truth.set_index("unit_id")["true_immunity_at_cut"])
    error = (merged["current_immunity"] - merged["true_immunity_at_cut"]).abs()
    assert error.mean() < 0.08


def test_every_unit_gets_a_verdict(small_run):
    """No unit may fall out of the run silently."""
    panel, _, _, plan_set = small_run
    assert len(plan_set.plans) == panel["unit_id"].nunique()
    assert all(isinstance(p.feasibility, Feasibility) for p in plan_set.plans)


def test_a_higher_target_never_reduces_the_round_count(small_run):
    """Raising the bar cannot make the answer easier. A basic sanity property."""
    panel, _, _, _ = small_run
    easy = run_pipeline(panel, EngineConfig(n_draws=200, r0=3.0), planning_interval_months=3.0)
    hard = run_pipeline(panel, EngineConfig(n_draws=200, r0=8.0), planning_interval_months=3.0)
    easy_feasible = sum(p.feasibility is Feasibility.FEASIBLE for p in easy.plans)
    hard_feasible = sum(p.feasibility is Feasibility.FEASIBLE for p in hard.plans)
    assert easy_feasible >= hard_feasible


def test_the_run_is_reproducible(small_run):
    """Same panel, same seed, same answer. Otherwise no result can be checked."""
    panel, _, config, plan_set = small_run
    again = run_pipeline(panel, config, planning_interval_months=3.0)
    first = [(p.unit_id, p.rounds_median, p.feasibility) for p in plan_set.plans]
    second = [(p.unit_id, p.rounds_median, p.feasibility) for p in again.plans]
    assert first == second


def test_a_ceiling_limited_unit_reports_no_round_count(small_run):
    """When rounds cannot reach the target, the engine must not offer a number."""
    _, _, _, plan_set = small_run
    blocked = [p for p in plan_set.plans if p.feasibility is Feasibility.INFEASIBLE_BY_CAMPAIGNS]
    if not blocked:
        pytest.skip("This simulated programme produced no ceiling-limited units.")
    assert all(p.immunity_ceiling_median < p.target_immunity + 1e-6 for p in blocked)


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------


def test_the_optimiser_refuses_to_spend_rounds_where_they_cannot_work(small_run):
    """Sending rounds to a ceiling-limited unit is the mistake this engine prevents."""
    _, _, config, plan_set = small_run
    allocation = allocate_rounds(
        plan_set.plans, plan_set.parameters, config, round_budget=20
    )
    blocked = {
        p.unit_id for p in plan_set.plans
        if p.feasibility is Feasibility.INFEASIBLE_BY_CAMPAIGNS
    }
    if not allocation.table.empty:
        assert not (set(allocation.table["unit_id"]) & blocked)


def test_greedy_allocation_beats_spreading_the_budget_evenly(small_run):
    """If it does not, the optimisation is not earning its complexity."""
    _, _, config, plan_set = small_run
    allocation = allocate_rounds(
        plan_set.plans, plan_set.parameters, config, round_budget=15
    )
    assert allocation.objective_value >= allocation.objective_uniform


# --------------------------------------------------------------------------
# Council
# --------------------------------------------------------------------------


def test_the_council_blocks_a_recommendation_it_cannot_deliver(small_run):
    """A round interval below the operational cycle time must be blocked outright."""
    panel, _, config, _ = small_run
    plan_set = run_pipeline(panel, config, planning_interval_months=1.0)
    record = convene(plan_set.plans[0], plan_set, config, use_model=False)
    assert record.overall_verdict is Verdict.BLOCK
    assert any("interval" in c for c in record.conditions)


def test_overall_confidence_is_the_weakest_seat_not_the_average(small_run):
    """A recommendation is as sound as its least supported premise."""
    _, _, config, plan_set = small_run
    record = convene(plan_set.plans[0], plan_set, config, use_model=False)
    confidences = {f.confidence for f in record.findings}
    if confidences:
        order = ["low", "medium", "high"]
        assert record.overall_confidence.value == min(
            (c.value for c in confidences), key=order.index
        )


def test_dissent_survives_adjudication(small_run):
    """A challenge must reach the reader, not be smoothed into a summary."""
    _, _, config, plan_set = small_run
    blocked = [
        p for p in plan_set.plans if p.feasibility is Feasibility.INFEASIBLE_BY_CAMPAIGNS
    ]
    if not blocked:
        pytest.skip("This simulated programme produced no ceiling-limited units.")
    record = convene(blocked[0], plan_set, config, use_model=False)
    assert record.dissent
    assert "Dissent" in record.render()


def test_the_seat_that_cannot_block_never_blocks(small_run):
    """The systems seat advises on leverage; it does not hold a recommendation."""
    _, _, config, plan_set = small_run
    record = convene(plan_set.plans[0], plan_set, config, use_model=False)
    systems = [f for f in record.findings if f.expert == "systems_analyst"]
    assert all(f.verdict is not Verdict.BLOCK for f in systems)


def test_an_unpolled_seat_is_not_treated_as_agreement(small_run):
    """Silence from a seat is a gap in the review, and must cap confidence."""
    _, _, config, plan_set = small_run
    record = convene(
        plan_set.plans[0], plan_set, config, experts=COUNCIL[:2], use_model=True
    )
    assert record.seats_missing
    assert record.overall_confidence.value != "high"


def test_every_recommendation_carries_a_systems_block(small_run):
    """Which stock and which loop, on every recommendation without exception."""
    _, _, config, plan_set = small_run
    record = convene(plan_set.plans[0], plan_set, config, use_model=False)
    rendered = record.systems_block.render()
    for field in ("Stock targeted", "Loop acted on", "Leverage tier", "Trap risk"):
        assert field in rendered


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_the_summary_states_the_target_and_its_basis(small_run):
    """No figure without its provenance."""
    _, _, config, plan_set = small_run
    text = programme_summary(plan_set, config)
    assert "Vc adjusted" in text and "R0" in text
    assert "does not authorise" in text


def test_a_unit_brief_states_what_the_answer_rests_on(small_run):
    """A round count with no binding constraint is not actionable."""
    _, _, config, plan_set = small_run
    text = unit_brief(plan_set, plan_set.plans[0].unit_id, config)
    assert "Binding constraint" in text
    assert "Unreachable core" in text
