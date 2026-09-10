"""Running the council and recording what it decided.

The council does not vote. Averaging six verdicts into one destroys the
information that made assembling six seats worthwhile, and it lets a confident
majority bury the one seat that happened to be looking at the failure that
matters. So the record keeps every verdict, keeps the reasons, and applies three
rules that are stated here rather than buried in a weighting scheme:

* One block inside a seat's own mandate holds the recommendation. The seats do
  not overlap, so a blocking seat is the only one looking at that failure mode.
* Overall confidence is the lowest confidence any seat reported, not the average.
  A chain is as strong as its weakest link and a recommendation is as sound as
  its least supported premise.
* Dissent is published with the recommendation, not filed behind it.

The evidence packet each seat sees is built once and shared, so two seats
disagreeing means they read the same numbers differently, which is a finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..contracts import EngineConfig, Feasibility
from ..pipeline import PlanSet
from ..rounds_required import UnitPlan
from .adjudication import SystemsBlock, adjudicate
from .experts import COUNCIL, Confidence, Expert, Verdict
from .providers import Finding, ModelReviewer
from .rules import RuleReviewer


@dataclass
class CouncilRecord:
    """What the council concluded about one recommendation.

    Attributes:
        unit_id: Unit reviewed.
        recommendation: The engine's plan, as reviewed.
        findings: Every finding from every seat, in the order the seats were polled.
        overall_verdict: The strongest objection raised.
        overall_confidence: The lowest confidence any seat reported.
        conditions: Binding conditions attached by seats that endorsed conditionally.
        dissent: Findings that challenge or block, kept separate so they cannot be
            skimmed past.
        missing_data: What each seat said would most raise its confidence. This is
            the engine's answer to "where should the next measurement go".
        systems_block: Which stock the recommendation moves and at what leverage.
        seats_polled: Seats that returned a review.
        seats_missing: Seats that could not be polled. Their failure modes went
            unreviewed and that is stated, never treated as assent.
    """

    unit_id: str
    recommendation: str
    findings: list[Finding]
    overall_verdict: Verdict
    overall_confidence: Confidence
    conditions: list[str]
    dissent: list[Finding]
    missing_data: list[str]
    systems_block: SystemsBlock
    seats_polled: list[str] = field(default_factory=list)
    seats_missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "recommendation": self.recommendation,
            "overall_verdict": self.overall_verdict.value,
            "overall_confidence": self.overall_confidence.value,
            "conditions": self.conditions,
            "missing_data": self.missing_data,
            "systems_block": self.systems_block.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "dissent": [f.to_dict() for f in self.dissent],
            "seats_polled": self.seats_polled,
            "seats_missing": self.seats_missing,
        }

    def render(self) -> str:
        """The record as a block that can go into a briefing unchanged."""
        lines = [
            f"COUNCIL RECORD — {self.unit_id}",
            f"Recommendation: {self.recommendation}",
            f"Verdict: {self.overall_verdict.value.replace('_', ' ')} "
            f"(confidence: {self.overall_confidence.value})",
            "",
        ]
        if self.conditions:
            lines.append("Binding conditions:")
            lines.extend(f"  {i}. {c}" for i, c in enumerate(self.conditions, 1))
            lines.append("")
        if self.dissent:
            lines.append("Dissent (not resolved, recorded):")
            for finding in self.dissent:
                lines.append(f"  [{finding.expert}] {finding.summary}")
            lines.append("")
        if self.missing_data:
            lines.append("What would most raise confidence:")
            lines.extend(f"  - {d}" for d in self.missing_data)
            lines.append("")
        if self.seats_missing:
            lines.append(
                "Seats not polled, whose failure modes went unreviewed: "
                + ", ".join(self.seats_missing)
            )
            lines.append("")
        lines.append(self.systems_block.render())
        return "\n".join(lines)


def build_packet(
    plan: UnitPlan,
    plan_set: PlanSet,
    config: EngineConfig,
) -> dict[str, Any]:
    """Assemble the evidence every seat reviews.

    The packet holds the answer *and* the machinery that produced it, because a
    seat cannot check a recommendation it can only see the conclusion of.

    Args:
        plan: The unit's plan.
        plan_set: The run the plan came from.
        config: Engine configuration.

    Returns:
        A JSON-serialisable packet.
    """
    parameters = (
        plan_set.parameters.loc[plan.unit_id].to_dict()
        if plan.unit_id in plan_set.parameters.index
        else {}
    )
    report = plan_set.reach_report

    return {
        "unit": {
            "unit_id": plan.unit_id,
            "grain": plan.grain,
            "lga_code": plan.lga_code,
            "state_code": plan.state_code,
            "target_population": _clean(parameters.get("target_pop")),
            "denominator_basis": parameters.get("denominator_basis"),
        },
        "recommendation": {
            "feasibility": plan.feasibility.value,
            "rounds_median": plan.rounds_median,
            "rounds_for_assurance": plan.rounds_assured,
            "assurance_level": config.assurance,
            "rounds_interval_80": list(plan.rounds_interval_80),
            "interval_months": plan.interval_months,
            "probability_by_round": plan.probability_by_round,
        },
        "immunity": {
            "target": plan.target_immunity,
            "target_basis": f"Vc(adj) at R0 {config.r0}, schedule failure {config.schedule_failure}",
            "current_estimated": plan.current_immunity,
            "current_estimate_sd": _clean(parameters.get("current_immunity_sd")),
            "schedule_ceiling": plan.immunity_ceiling_median,
            "core_limited_ceiling": plan.core_ceiling_median,
            "trough_between_rounds": plan.immunity_trough_median,
            "holding_interval_months": plan.holding_interval_months,
            "probability_ceiling_below_target": plan.probability_infeasible,
            "fatigue_risk": plan.fatigue_risk,
            "probability_short_if_reach_keeps_falling": plan.probability_infeasible_with_drift,
        },
        "reach_parameters": {
            "unreachable_core_share": _clean(parameters.get("pi_zero")),
            "reach_concentration_kappa": _clean(parameters.get("kappa")),
            "round_to_round_stickiness_rho": _clean(parameters.get("rho")),
            "reach_drift_per_round": _clean(parameters.get("reach_drift_per_round")),
            "denominator_inflation_assumed": _clean(parameters.get("denominator_inflation")),
            "routine_immunisation_protection": _clean(parameters.get("ri_protection")),
            "births_per_month": _clean(parameters.get("births_per_month")),
        },
        "model_quality": {
            "reach_mae": _clean(report.mae) if report else None,
            "reach_skill_over_persistence": _clean(report.skill_vs_persistence) if report else None,
            "reach_interval_coverage": report.interval_coverage if report else None,
            "calibration_verdict": report.calibration_verdict() if report else "Reach model not fitted.",
            "reach_model_notes": report.notes if report else ["Reach model not fitted."],
            "top_predictors": list(report.feature_importance.items())[:5] if report else [],
        },
        "binding_constraint": plan.binding_constraint,
        "provenance": plan_set.provenance.entries,
        "programme_notes": plan_set.notes,
    }


def _clean(value: Any) -> Any:
    """Make a value JSON-safe, turning non-finite floats into None."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def convene(
    plan: UnitPlan,
    plan_set: PlanSet,
    config: EngineConfig,
    *,
    experts: tuple[Expert, ...] = COUNCIL,
    use_model: bool = True,
    model_reviewer: ModelReviewer | None = None,
) -> CouncilRecord:
    """Poll the council on one unit's recommendation.

    The deterministic rule reviewer always runs. The language-model reviewer runs
    in addition when it is available, and its absence is recorded rather than
    passed over, because a seat that was not polled has not agreed.

    Args:
        plan: The unit's plan.
        plan_set: The run the plan came from.
        config: Engine configuration.
        experts: Seats to poll.
        use_model: Whether to attempt the language-model pass.
        model_reviewer: Reviewer to use. One is constructed from the environment
            when omitted.

    Returns:
        A ``CouncilRecord``.
    """
    packet = build_packet(plan, plan_set, config)

    rules = RuleReviewer(config)
    findings: list[Finding] = []
    for expert in experts:
        findings.extend(rules.review(expert, packet))

    seats_polled = [e.key for e in experts]
    seats_missing: list[str] = []

    if use_model:
        reviewer = model_reviewer or ModelReviewer()
        if reviewer.available:
            for expert in experts:
                model_findings = reviewer.review(expert, packet)
                if model_findings:
                    findings.extend(model_findings)
                else:
                    seats_missing.append(f"{expert.key} (model pass)")
        else:
            seats_missing.append(
                "all seats (model pass unavailable; deterministic rules only)"
            )

    return adjudicate(
        unit_id=plan.unit_id,
        recommendation=plan.headline(),
        findings=findings,
        plan=plan,
        packet=packet,
        seats_polled=seats_polled,
        seats_missing=seats_missing,
    )


def convene_many(
    plan_set: PlanSet,
    config: EngineConfig,
    *,
    limit: int | None = None,
    use_model: bool = False,
) -> list[CouncilRecord]:
    """Poll the council on the units where the recommendation matters most.

    Reviewing every settlement is neither affordable nor useful. The units that
    get reviewed are those whose verdict would change what a programme does:
    everything the engine says campaigns cannot fix, then everything flagged for
    fatigue, then the largest remaining populations.

    Args:
        plan_set: A finished run.
        config: Engine configuration.
        limit: Maximum units to review. ``None`` reviews all of them.
        use_model: Whether to attempt the language-model pass. Defaults to False
            because a national run would otherwise make thousands of calls.

    Returns:
        Records in review order.
    """
    def priority(plan: UnitPlan) -> tuple[int, float]:
        if plan.feasibility is Feasibility.INFEASIBLE_BY_CAMPAIGNS:
            rank = 0
        elif plan.fatigue_risk:
            rank = 1
        elif plan.feasibility is Feasibility.INSUFFICIENT_DATA:
            rank = 2
        else:
            rank = 3
        size = 0.0
        if plan.unit_id in plan_set.parameters.index:
            size = float(plan_set.parameters.loc[plan.unit_id].get("target_pop", 0.0) or 0.0)
        return (rank, -size)

    ordered = sorted(plan_set.plans, key=priority)
    if limit is not None:
        ordered = ordered[:limit]
    return [convene(plan, plan_set, config, use_model=use_model) for plan in ordered]
