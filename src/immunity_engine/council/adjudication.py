"""Turning six independent reviews into one record, without averaging them away.

Adjudication here is deliberately mechanical. The rules are stated, applied the
same way every time, and produce a record a reader can re-derive from the
findings. A weighted score would be smoother and would hide exactly the thing a
council exists to surface, which is that one seat saw something the others did not.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from ..contracts import Feasibility
from ..rounds_required import UnitPlan
from .experts import COUNCIL_BY_KEY, Confidence, Verdict
from .providers import Finding

#: Verdicts in order of how much they hold up a recommendation.
_SEVERITY: dict[Verdict, int] = {
    Verdict.ENDORSE: 0,
    Verdict.ENDORSE_WITH_CONDITIONS: 1,
    Verdict.CHALLENGE: 2,
    Verdict.BLOCK: 3,
}

_CONFIDENCE_ORDER: dict[Confidence, int] = {
    Confidence.LOW: 0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH: 2,
}


@dataclass(frozen=True)
class SystemsBlock:
    """What the recommendation acts on, in systems terms.

    Attached to every recommendation because the same round count means different
    things depending on which loop it touches. A plan that strengthens a delayed
    balancing loop and a plan that weakens a reinforcing one are not comparable,
    and a reader given only "four rounds" cannot tell them apart.

    Attributes:
        stock: What the recommendation moves.
        loop: The feedback loop it acts on.
        direction: Whether it strengthens a balancing loop, weakens a reinforcing
            one, or neither.
        delay: The delay in that loop, and where the figure comes from.
        leverage_tier: Meadows leverage tier, 1 highest and 12 lowest.
        trap_risk: The named systems trap this recommendation could fall into.
    """

    stock: str
    loop: str
    direction: str
    delay: str
    leverage_tier: int
    trap_risk: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        return "\n".join(
            [
                "SYSTEMS BLOCK",
                f"  Stock targeted:     {self.stock}",
                f"  Loop acted on:      {self.loop}",
                f"  Loop direction:     {self.direction}",
                f"  Delay in this loop: {self.delay}",
                f"  Leverage tier:      {self.leverage_tier}",
                f"  Trap risk:          {self.trap_risk}",
                (
                    "  Note: leverage tier 10 or lower with direction 'neither' means this is a "
                    "parameter adjustment, not a strategy."
                    if self.leverage_tier >= 10 and self.direction == "neither"
                    else ""
                ),
            ]
        ).rstrip()


def build_systems_block(plan: UnitPlan, packet: dict[str, Any]) -> SystemsBlock:
    """Derive the systems block from the recommendation itself.

    Args:
        plan: The unit's plan.
        packet: The evidence packet the council reviewed.

    Returns:
        A ``SystemsBlock``.
    """
    interval = plan.interval_months
    holding = plan.holding_interval_months

    if plan.feasibility is Feasibility.INFEASIBLE_BY_CAMPAIGNS:
        core_binds = plan.core_ceiling_median < plan.target_immunity + 0.005
        return SystemsBlock(
            stock=f"Susceptible children in {plan.unit_id}, capped by an unreachable share",
            loop=(
                "R: low reach to sustained transmission to more rounds to fatigue to lower reach"
                if not core_binds
                else "B: campaign to immunity, structurally unable to close"
            ),
            direction="weaken reinforcing" if not core_binds else "neither",
            delay=(
                f"Immunity settles at {plan.immunity_ceiling_median:.1%} against a target of "
                f"{plan.target_immunity:.1%}"
            ),
            leverage_tier=5 if core_binds else 7,
            trap_risk=(
                "Success to the successful: rounds keep going to units that report well, "
                "while the children this unit cannot reach stay unmeasured and unreached."
            ),
        )

    if plan.fatigue_risk:
        return SystemsBlock(
            stock=f"Susceptible children in {plan.unit_id}, and team and community goodwill",
            loop="R: falling reach to more susceptibles to more rounds to more fatigue",
            direction="weaken reinforcing",
            delay=f"Reach declines round on round; rounds are {interval:.0f} months apart",
            leverage_tier=7,
            trap_risk=(
                "Drift to low performance: each round is benchmarked against the last one "
                "rather than against the immunity target, so decline reads as normal."
            ),
        )

    oscillating = holding is not None and holding < interval
    return SystemsBlock(
        stock=f"Susceptible children under five in {plan.unit_id}",
        loop="B: campaign rounds to immunity to fewer susceptibles",
        direction="strengthen balancing",
        delay=(
            f"Births refill the stock continuously; immunity falls to "
            f"{plan.immunity_trough_median:.1%} between rounds {interval:.0f} months apart"
        ),
        leverage_tier=8,
        trap_risk=(
            "Oscillation: the round interval is longer than the interval that holds immunity, "
            f"so protection peaks and falls back rather than holding."
            if oscillating
            else "Rule beating: a round count becomes a target that is reported met rather than achieved."
        ),
    )


def adjudicate(
    *,
    unit_id: str,
    recommendation: str,
    findings: list[Finding],
    plan: UnitPlan,
    packet: dict[str, Any],
    seats_polled: list[str],
    seats_missing: list[str],
) -> "CouncilRecord":
    """Combine findings into a record.

    Three rules, applied in this order:

    1. The overall verdict is the most severe verdict any seat reached inside its
       own mandate. A block is not outvoted, because no other seat was looking at
       that failure mode.
    2. Overall confidence is the lowest confidence any seat reported. A
       recommendation is as sound as its least supported premise.
    3. Dissent is separated out and published, never merged into the summary.

    Args:
        unit_id: Unit reviewed.
        recommendation: The engine's headline, as reviewed.
        findings: Every finding from every seat.
        plan: The unit's plan.
        packet: The evidence packet.
        seats_polled: Seats that returned a review.
        seats_missing: Seats that could not be polled.

    Returns:
        A ``CouncilRecord``.
    """
    from .council import CouncilRecord  # imported here to avoid a circular import

    blocking = [
        f for f in findings
        if f.verdict is Verdict.BLOCK and COUNCIL_BY_KEY.get(f.expert, None) is not None
        and COUNCIL_BY_KEY[f.expert].may_block
    ]
    if blocking:
        overall = Verdict.BLOCK
    elif findings:
        overall = max((f.verdict for f in findings), key=lambda v: _SEVERITY[v])
    else:
        overall = Verdict.ENDORSE

    if findings:
        confidence = min(
            (f.confidence for f in findings), key=lambda c: _CONFIDENCE_ORDER[c]
        )
    else:
        confidence = Confidence.LOW

    # A seat that was not polled did not agree. Unreviewed mandates cap confidence.
    if seats_missing and confidence is Confidence.HIGH:
        confidence = Confidence.MEDIUM

    conditions = [f.condition for f in findings if f.condition]
    # Preserve order while removing repeats, so the same condition raised by two
    # seats appears once but keeps its position.
    conditions = list(dict.fromkeys(conditions))

    dissent = [f for f in findings if f.verdict in (Verdict.CHALLENGE, Verdict.BLOCK)]
    missing = list(dict.fromkeys(f.missing_datum for f in findings if f.missing_datum))

    return CouncilRecord(
        unit_id=unit_id,
        recommendation=recommendation,
        findings=findings,
        overall_verdict=overall,
        overall_confidence=confidence,
        conditions=conditions,
        dissent=dissent,
        missing_data=missing,
        systems_block=build_systems_block(plan, packet),
        seats_polled=seats_polled,
        seats_missing=seats_missing,
    )
