"""The deterministic half of the council.

Each check below is a threshold this programme has been caught by before,
written as code so it gives the same answer every time and can be argued with
line by line. Nothing here needs a network, a key or a model, which means the
safety findings of a run never depend on a service being up.

The rules are conservative in one specific direction: they raise a finding when
the evidence is thin, rather than staying silent. A council seat that says
nothing reads as assent, and the most expensive failures in immunisation
analytics have been assumptions nobody wrote down.
"""

from __future__ import annotations

from typing import Any

from ..contracts import EngineConfig
from .experts import Confidence, Expert, Verdict
from .providers import Finding


class RuleReviewer:
    """Applies each seat's deterministic checks to an evidence packet."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def review(self, expert: Expert, packet: dict[str, Any]) -> list[Finding]:
        """Return the findings this seat's rules raise.

        Args:
            expert: The seat.
            packet: Evidence packet from ``council.build_packet``.

        Returns:
            Findings, possibly empty. An empty list means every rule passed, not
            that the seat has nothing to add.
        """
        handler = getattr(self, f"_{expert.key}", None)
        if handler is None:
            return []
        findings = handler(packet)
        if not expert.may_block:
            findings = [
                f if f.verdict is not Verdict.BLOCK
                else Finding(**{**f.__dict__, "verdict": Verdict.CHALLENGE})
                for f in findings
            ]
        return findings

    # -- seats -------------------------------------------------------------

    def _epidemiologist(self, packet: dict[str, Any]) -> list[Finding]:
        immunity = packet["immunity"]
        reach = packet["reach_parameters"]
        findings: list[Finding] = []

        gap = immunity["target"] - (immunity["current_estimated"] or 0.0)
        if gap > 0.40:
            findings.append(
                Finding(
                    expert="epidemiologist",
                    verdict=Verdict.CHALLENGE,
                    confidence=Confidence.MEDIUM,
                    summary=(
                        f"The gap to target is {gap:.0%}, which is larger than any single "
                        "outbreak response has closed here. A plan that closes it in the "
                        "stated number of rounds should be read as a best case."
                    ),
                    evidence={
                        "current_immunity": immunity["current_estimated"],
                        "target": immunity["target"],
                    },
                    missing_datum="A serosurvey or an age-stratified immunity estimate for this unit.",
                )
            )

        if immunity["holding_interval_months"] is None:
            findings.append(
                Finding(
                    expert="epidemiologist",
                    verdict=Verdict.CHALLENGE,
                    confidence=Confidence.MEDIUM,
                    summary=(
                        "No tested round interval holds immunity above the threshold between "
                        "rounds. Reaching the target once would leave a window that reopens "
                        "before the next round, so transmission is interrupted only briefly."
                    ),
                    evidence={"trough_between_rounds": immunity["trough_between_rounds"]},
                    condition=(
                        "Pair the campaign plan with a routine-immunisation target for this "
                        "unit; campaigns at any practical frequency will not hold the level alone."
                    ),
                )
            )

        stickiness = reach.get("round_to_round_stickiness_rho")
        if stickiness is not None and stickiness > 0.75:
            findings.append(
                Finding(
                    expert="epidemiologist",
                    verdict=Verdict.ENDORSE_WITH_CONDITIONS,
                    confidence=Confidence.MEDIUM,
                    summary=(
                        f"Reach is highly persistent (rho {stickiness:.2f}), so the children "
                        "still susceptible are largely the same children each round. "
                        "Population immunity will overstate protection where those children "
                        "live together."
                    ),
                    evidence={"rho": stickiness},
                    condition=(
                        "Check whether remaining susceptibles cluster geographically before "
                        "treating the unit-level figure as sufficient."
                    ),
                )
            )
        return findings

    def _programme_manager(self, packet: dict[str, Any]) -> list[Finding]:
        recommendation = packet["recommendation"]
        reach = packet["reach_parameters"]
        findings: list[Finding] = []

        interval = recommendation["interval_months"]
        if interval < 2.0:
            findings.append(
                Finding(
                    expert="programme_manager",
                    verdict=Verdict.BLOCK,
                    confidence=Confidence.HIGH,
                    summary=(
                        f"A {interval:.1f}-month interval is shorter than the cycle time for "
                        "microplanning, team training and vaccine arrival. The plan cannot be "
                        "executed as scheduled."
                    ),
                    evidence={"interval_months": interval},
                    condition="Re-plan at an interval the supply and workforce cycle supports.",
                )
            )

        rounds = recommendation["rounds_for_assurance"]
        if rounds is not None and rounds >= 6:
            findings.append(
                Finding(
                    expert="programme_manager",
                    verdict=Verdict.CHALLENGE,
                    confidence=Confidence.MEDIUM,
                    summary=(
                        f"{rounds} rounds in one unit runs past what has been observed here. "
                        "Community and team fatigue beyond about six rounds is outside the "
                        "range the reach model was fitted on, so later rounds are extrapolation."
                    ),
                    evidence={"rounds_for_assurance": rounds},
                    missing_datum="Reach measured in a unit that has run more than six consecutive rounds.",
                )
            )

        drift = reach.get("reach_drift_per_round")
        if drift is not None and drift < -0.01:
            findings.append(
                Finding(
                    expert="programme_manager",
                    verdict=Verdict.ENDORSE_WITH_CONDITIONS,
                    confidence=Confidence.MEDIUM,
                    summary=(
                        f"Reach is falling by {abs(drift):.1%} per round. The round count "
                        "assumes that continues; holding reach flat would shorten the plan."
                    ),
                    evidence={"reach_drift_per_round": drift},
                    condition=(
                        "Attach a reach floor to the plan and measure it after each round, "
                        "rather than treating the decline as fixed."
                    ),
                )
            )
        return findings

    def _statistician(self, packet: dict[str, Any]) -> list[Finding]:
        quality = packet["model_quality"]
        recommendation = packet["recommendation"]
        findings: list[Finding] = []

        coverage = quality.get("reach_interval_coverage") or {}
        miscalibrated = {
            name: value
            for name, value in coverage.items()
            if abs(value - float(name.rstrip("%")) / 100.0) > 0.07
        }
        if miscalibrated:
            findings.append(
                Finding(
                    expert="statistician",
                    verdict=Verdict.ENDORSE_WITH_CONDITIONS,
                    confidence=Confidence.MEDIUM,
                    summary=(
                        "The reach model's prediction intervals are not calibrated, so the "
                        "confidence attached to the round count does not mean what it says."
                    ),
                    evidence={"interval_coverage": miscalibrated},
                    condition=(
                        "Report the round count as a range without a confidence level until "
                        "interval coverage is within tolerance."
                    ),
                    missing_datum="More held-out rounds, to measure coverage precisely.",
                )
            )

        skill = quality.get("reach_skill_over_persistence")
        if skill is not None and skill <= 0.05:
            findings.append(
                Finding(
                    expert="statistician",
                    verdict=Verdict.CHALLENGE,
                    confidence=Confidence.HIGH,
                    summary=(
                        f"The reach model beats assuming next round repeats last round by "
                        f"{skill:.0%}. The learned layer is not earning its place here."
                    ),
                    evidence={"skill_over_persistence": skill},
                    condition="Use the persistence baseline and state that no model was used.",
                )
            )

        low, high = recommendation["rounds_interval_80"]
        if low is not None and high is not None and high - low >= 4:
            findings.append(
                Finding(
                    expert="statistician",
                    verdict=Verdict.ENDORSE_WITH_CONDITIONS,
                    confidence=Confidence.MEDIUM,
                    summary=(
                        f"The 80% interval spans {low} to {high} rounds. Quoting the median "
                        "alone would imply a precision the evidence does not carry."
                    ),
                    evidence={"rounds_interval_80": [low, high]},
                    condition="Publish the interval wherever the round count is published.",
                )
            )

        if packet["unit"]["grain"] != "lga":
            findings.append(
                Finding(
                    expert="statistician",
                    verdict=Verdict.ENDORSE_WITH_CONDITIONS,
                    confidence=Confidence.MEDIUM,
                    summary=(
                        f"This is a {packet['unit']['grain']}-grain figure. It is valid for "
                        "targeting, but inference belongs at LGA level, where the "
                        "epidemiologically relevant variation sits."
                    ),
                    evidence={"grain": packet["unit"]["grain"]},
                    condition=(
                        "Carry the LGA distribution alongside any sub-LGA figure that is "
                        "quoted upward."
                    ),
                )
            )
        return findings

    def _access_analyst(self, packet: dict[str, Any]) -> list[Finding]:
        immunity = packet["immunity"]
        reach = packet["reach_parameters"]
        findings: list[Finding] = []

        core = reach.get("unreachable_core_share")
        if core is not None and core > 0.05:
            findings.append(
                Finding(
                    expert="access_analyst",
                    verdict=Verdict.CHALLENGE,
                    confidence=Confidence.HIGH,
                    summary=(
                        f"{core:.1%} of children in this unit are estimated to be reachable in "
                        "no round at all. That share caps immunity regardless of round count."
                    ),
                    evidence={
                        "unreachable_core": core,
                        "core_limited_ceiling": immunity["core_limited_ceiling"],
                    },
                    condition=(
                        "Fund an access or enumeration action alongside any rounds, and size "
                        "it against this share."
                    ),
                    missing_datum="An enumeration of children in settlements the microplan does not list.",
                )
            )

        if immunity["probability_ceiling_below_target"] and immunity["probability_ceiling_below_target"] > 0.5:
            findings.append(
                Finding(
                    expert="access_analyst",
                    verdict=Verdict.BLOCK,
                    confidence=Confidence.HIGH,
                    summary=(
                        "The immunity ceiling sits below the target in the majority of draws. "
                        "Allocating rounds here spends them on a target they cannot reach."
                    ),
                    evidence={
                        "probability_ceiling_below_target": immunity["probability_ceiling_below_target"],
                        "schedule_ceiling": immunity["schedule_ceiling"],
                    },
                    condition=(
                        "Route this unit to the access and denominator workstream. Rounds may "
                        "still be justified to suppress transmission, but not on the grounds "
                        "of reaching this target."
                    ),
                )
            )
        return findings

    def _systems_analyst(self, packet: dict[str, Any]) -> list[Finding]:
        immunity = packet["immunity"]
        findings: list[Finding] = []

        if immunity["holding_interval_months"] is None and immunity["probability_ceiling_below_target"] is not None:
            findings.append(
                Finding(
                    expert="systems_analyst",
                    verdict=Verdict.CHALLENGE,
                    confidence=Confidence.MEDIUM,
                    summary=(
                        "Adding rounds strengthens a balancing loop whose delay is longer than "
                        "the interval between rounds. That produces oscillation, not control: "
                        "immunity peaks after each round and falls back before the next."
                    ),
                    evidence={
                        "trough_between_rounds": immunity["trough_between_rounds"],
                        "peak_ceiling": immunity["schedule_ceiling"],
                    },
                    condition=(
                        "Compare the marginal value of one more round against reducing the "
                        "detection-to-response delay, which acts on the same loop at higher leverage."
                    ),
                )
            )

        if immunity["fatigue_risk"]:
            findings.append(
                Finding(
                    expert="systems_analyst",
                    verdict=Verdict.CHALLENGE,
                    confidence=Confidence.MEDIUM,
                    summary=(
                        "Reach falling round on round is a reinforcing loop: lower reach means "
                        "more susceptibles, more rounds, more fatigue, lower reach. Adding "
                        "rounds feeds it. Weakening it beats strengthening the response."
                    ),
                    evidence={
                        "probability_short_if_reach_keeps_falling": immunity[
                            "probability_short_if_reach_keeps_falling"
                        ]
                    },
                    condition="Name what will stop reach falling before adding rounds to the calendar.",
                )
            )
        return findings

    def _data_auditor(self, packet: dict[str, Any]) -> list[Finding]:
        reach = packet["reach_parameters"]
        unit = packet["unit"]
        findings: list[Finding] = []

        inflation = reach.get("denominator_inflation_assumed")
        if inflation is not None and inflation > 1.05:
            findings.append(
                Finding(
                    expert="data_auditor",
                    verdict=Verdict.CHALLENGE,
                    confidence=Confidence.HIGH,
                    summary=(
                        f"The denominator is treated as inflated by {inflation:.2f}x, inferred "
                        "from administrative coverage above 100%. The round count moves with "
                        "that factor, and the factor is an inference, not a measurement."
                    ),
                    evidence={
                        "denominator_inflation": inflation,
                        "denominator_basis": unit.get("denominator_basis"),
                    },
                    condition=(
                        "Reconcile the microplan target against the last enumeration before "
                        "the first round."
                    ),
                    missing_datum="A recount of the target population under a stated basis.",
                )
            )

        if not unit.get("denominator_basis"):
            findings.append(
                Finding(
                    expert="data_auditor",
                    verdict=Verdict.BLOCK,
                    confidence=Confidence.HIGH,
                    summary=(
                        "No denominator provenance is attached to this unit's target "
                        "population, so its coverage figures cannot be compared with anything."
                    ),
                    evidence={"denominator_basis": None},
                    condition="Attach a denominator basis before the figure is used.",
                )
            )

        for entry in packet.get("provenance", []):
            if entry.get("field") == "verified_vaccinated":
                findings.append(
                    Finding(
                        expert="data_auditor",
                        verdict=Verdict.CHALLENGE,
                        confidence=Confidence.HIGH,
                        summary=(
                            "Reach rests partly on administrative counts rather than independent "
                            "verification. Where the two diverge, this round count is optimistic."
                        ),
                        evidence={"assumption": entry.get("assumption")},
                        condition="Verify coverage independently in this unit before committing rounds.",
                        missing_datum="An independent verification sample for the most recent round.",
                    )
                )
                break
        return findings
