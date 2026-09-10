"""Turning a run into something a decision maker can read and check.

The rule this module follows: no number appears without what it rests on. A round
count with no interval, no denominator basis and no statement of what would change
it is not a finding, it is a rumour with a decimal point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import EngineConfig, Feasibility
from .pipeline import PlanSet


def programme_summary(plan_set: PlanSet, config: EngineConfig) -> str:
    """The national picture: what the run found, and how much of it to believe.

    Args:
        plan_set: A finished run.
        config: Engine configuration.

    Returns:
        A text block.
    """
    frame = plan_set.to_frame()
    if frame.empty:
        return "No units were planned."

    counts = frame["feasibility"].value_counts()
    total = len(frame)
    reconciliation = config.reconcile_take()

    lines = [
        "CAMPAIGN REQUIREMENT — PROGRAMME SUMMARY",
        "=" * 72,
        "",
        "TARGET",
        f"  Immunity threshold      {config.vc_adjusted:.1%}  "
        f"(Vc adjusted, R0 {config.r0}, schedule failure {config.schedule_failure})",
        f"  Per-dose take used      {config.per_dose_take_mean:.3f}",
        f"  Implied by the stated schedule failure over {config.doses_in_schedule} doses: "
        f"{reconciliation['per_dose_take_implied_by_epsilon']:.3f}",
        f"  Residual between the two: {reconciliation['residual_pp']:+.2f} pp of schedule efficacy",
        "",
        "VERDICTS",
    ]
    for verdict in Feasibility:
        n = int(counts.get(verdict.value, 0))
        if n:
            lines.append(f"  {n:>6}  ({n / total:>5.1%})  {verdict.value.replace('_', ' ')}")

    fatigue = int(frame["fatigue_risk"].sum()) if "fatigue_risk" in frame else 0
    if fatigue:
        lines.append(
            f"  {fatigue:>6}  ({fatigue / total:>5.1%})  additionally flagged: reachable only if "
            "reach stops falling"
        )

    feasible = frame[frame["feasibility"] == Feasibility.FEASIBLE.value]
    if not feasible.empty:
        lines += [
            "",
            "ROUNDS NEEDED, WHERE CAMPAIGNS CAN GET THERE",
            f"  Median across units     {feasible['rounds_median'].median():.0f}",
            f"  For {config.assurance:.0%} assurance      "
            f"{feasible['rounds_assured'].median():.0f} (median across units)",
            f"  Total rounds if every feasible unit is planned to assurance: "
            f"{int(feasible['rounds_assured'].sum())}",
        ]

    blocked = frame[frame["feasibility"] == Feasibility.INFEASIBLE_BY_CAMPAIGNS.value]
    if not blocked.empty:
        core_bound = int((blocked["core_ceiling_median"] < blocked["target_immunity"]).sum())
        lines += [
            "",
            "WHERE ROUNDS ARE THE WRONG INSTRUMENT",
            f"  {len(blocked)} units cannot reach the target at any round count.",
            f"  {core_bound} are capped by children no round reaches: an access and "
            "enumeration problem.",
            f"  {len(blocked) - core_bound} are capped by births refilling the stock between "
            "rounds: an interval and routine-immunisation problem.",
            f"  Median ceiling in this group: {blocked['immunity_ceiling_median'].median():.1%} "
            f"against a target of {config.vc_adjusted:.1%}.",
        ]

    if "binding_constraint" in frame.columns:
        # The stored constraint carries its correlation, which makes every string
        # unique. Group on the label so the counts mean something.
        labels = frame["binding_constraint"].str.split(" — ").str[0]
        top = labels.value_counts().head(3)
        lines += ["", "WHAT WOULD SHARPEN THESE ANSWERS MOST"]
        lines += [f"  {n:>6} units  {constraint}" for constraint, n in top.items()]

    report = plan_set.reach_report
    lines += ["", "HOW MUCH TO BELIEVE THE ABOVE"]
    if report is None:
        lines.append(
            "  The reach model was not fitted. Round counts rest on each unit's recent "
            "history alone and should be read as order-of-magnitude."
        )
    else:
        lines += [
            f"  Reach forecast error (out of sample)  {report.mae:.3f} reach points",
            f"  Skill over assuming last round repeats {report.skill_vs_persistence:+.1%}",
            f"  Interval coverage                     {report.interval_coverage}",
            f"  {report.calibration_verdict()}",
        ]
        lines += [f"  Note: {n}" for n in report.notes]

    if plan_set.provenance.entries:
        lines += ["", "ASSUMPTIONS THAT WERE NECESSARY"]
        for entry in plan_set.provenance.entries:
            lines.append(f"  {entry['field']}: {entry['assumption']}")
            lines.append(f"    Consequence: {entry['impact']}")

    lines += ["", "PARAMETERS ESTIMATED FROM THE PANEL"]
    lines += [f"  {note}" for note in plan_set.notes]

    lines += [
        "",
        "This engine produces advice. It does not authorise a classification or a "
        "campaign calendar; a named human approver does.",
    ]
    return "\n".join(lines)


def unit_brief(plan_set: PlanSet, unit_id: str, config: EngineConfig) -> str:
    """A one-unit brief, with the answer and everything it depends on.

    Args:
        plan_set: A finished run.
        unit_id: Unit to brief on.
        config: Engine configuration.

    Returns:
        A text block.

    Raises:
        KeyError: If the unit is not in the run.
    """
    plan = next((p for p in plan_set.plans if p.unit_id == unit_id), None)
    if plan is None:
        raise KeyError(f"Unit '{unit_id}' is not in this run.")

    parameters = (
        plan_set.parameters.loc[unit_id] if unit_id in plan_set.parameters.index else None
    )

    lines = [
        f"CAMPAIGN REQUIREMENT — {unit_id}",
        "=" * 72,
        "",
        plan.headline(),
        "",
        "IMMUNITY",
        f"  Now                       {plan.current_immunity:.1%}",
        f"  Target                    {plan.target_immunity:.1%}",
        f"  Ceiling at this schedule  {plan.immunity_ceiling_median:.1%}",
        f"  Ceiling if only the unreachable core bound it: {plan.core_ceiling_median:.1%}",
        f"  Between rounds it falls to {plan.immunity_trough_median:.1%}",
        f"  Longest interval that holds the target: "
        + (
            f"{plan.holding_interval_months:.0f} months"
            if plan.holding_interval_months
            else "none of the intervals tested holds it"
        ),
        "",
        "ROUNDS",
        f"  Plan tempo                every {plan.interval_months:.0f} months",
    ]

    if plan.probability_by_round:
        lines.append("  Chance each round count is enough:")
        for n, probability in sorted(plan.probability_by_round.items()):
            if n > 8 and probability > 0.995:
                break
            bar = "#" * int(round(probability * 30))
            lines.append(f"    {n:>2} rounds  {probability:>5.0%}  {bar}")

    lines += [
        "",
        "WHAT THIS RESTS ON",
        f"  Binding constraint: {plan.binding_constraint}",
    ]
    if parameters is not None:
        lines += [
            f"  Unreachable core          {parameters.get('pi_zero', float('nan')):.1%} of children",
            f"  Reach stickiness (rho)    {parameters.get('rho', float('nan')):.2f}",
            f"  Reach spread (kappa)      {parameters.get('kappa', float('nan')):.1f}",
            f"  Reach drift per round     {parameters.get('reach_drift_per_round', 0.0):+.1%}",
            f"  Denominator inflation     {parameters.get('denominator_inflation', 1.0):.2f}x "
            f"({parameters.get('denominator_basis', 'basis not stated')})",
            f"  Routine immunisation      {parameters.get('ri_protection', float('nan')):.0%} "
            "of the birth cohort",
        ]
    if plan.fatigue_risk:
        lines += [
            "",
            f"  Reach is falling. If that continues, the target is missed in "
            f"{plan.probability_infeasible_with_drift:.0%} of draws even though it is "
            "structurally reachable.",
        ]
    return "\n".join(lines)


def escalation_table(plan_set: PlanSet) -> pd.DataFrame:
    """Units where a round is not the right thing to send, ordered by size of gap.

    Args:
        plan_set: A finished run.

    Returns:
        A frame ready to hand to whoever owns access and denominators.
    """
    frame = plan_set.escalations()
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["gap_to_target_pp"] = 100.0 * (frame["target_immunity"] - frame["immunity_ceiling_median"])
    frame["capped_by"] = np.where(
        frame["core_ceiling_median"] < frame["target_immunity"],
        "children no round reaches",
        "births refilling the stock between rounds",
    )
    columns = [
        "unit_id", "lga_code", "state_code", "current_immunity",
        "immunity_ceiling_median", "gap_to_target_pp", "capped_by", "binding_constraint",
    ]
    return frame[[c for c in columns if c in frame.columns]].sort_values(
        "gap_to_target_pp", ascending=False
    )
