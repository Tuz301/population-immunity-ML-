"""Command line entry point.

    python -m immunity_engine plan       --panel panel.csv --out plans.csv
    python -m immunity_engine allocate   --panel panel.csv --budget 40
    python -m immunity_engine council    --panel panel.csv --unit LGA1-2
    python -m immunity_engine validate                     # simulated programme
    python -m immunity_engine backtest   --panel panel.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .adapters.csvfile import load_csv_panel
from .adapters.synthetic import SyntheticProgramme, generate_synthetic_panel
from .contracts import EngineConfig
from .council import convene_many
from .optimizer import Objective, allocate_rounds
from .pipeline import run_pipeline
from .report import escalation_table, programme_summary, unit_brief
from .validation import backtest_on_panel, summarise_backtest, validate_against_truth


def _config(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        r0=args.r0,
        assurance=args.assurance,
        max_rounds=args.max_rounds,
        n_draws=args.draws,
    )


def _load(args: argparse.Namespace) -> pd.DataFrame:
    if args.panel:
        panel, report = load_csv_panel(args.panel)
        for warning in report.warnings:
            print(f"  warning: {warning}", file=sys.stderr)
        return panel
    print("  No panel given; using a simulated programme.", file=sys.stderr)
    panel, _ = generate_synthetic_panel(SyntheticProgramme())
    return panel


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--panel", type=Path, help="Panel file. Omit to use a simulated programme.")
    parser.add_argument("--r0", type=float, default=6.0, help="Basic reproduction number.")
    parser.add_argument("--assurance", type=float, default=0.90, help="Planning confidence level.")
    parser.add_argument("--max-rounds", dest="max_rounds", type=int, default=12, help="Round budget.")
    parser.add_argument("--draws", type=int, default=2000, help="Monte Carlo draws per unit.")
    parser.add_argument(
        "--interval", type=float, default=3.0, help="Planned months between rounds."
    )
    parser.add_argument(
        "--require-trough",
        action="store_true",
        help="Require immunity to hold above the target between rounds, not only after one.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="immunity_engine",
        description="How many campaign rounds reach population immunity, and where none will.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Round requirement per unit.")
    _add_common(plan)
    plan.add_argument("--out", type=Path, help="Write the plan table here as CSV.")
    plan.add_argument("--unit", help="Print a full brief for one unit.")

    allocate = subparsers.add_parser("allocate", help="Spread a round budget across units.")
    _add_common(allocate)
    allocate.add_argument("--budget", type=int, required=True, help="Rounds available.")
    allocate.add_argument(
        "--objective",
        choices=[o.value for o in Objective],
        default=Objective.SUSCEPTIBLES_AVERTED.value,
    )
    allocate.add_argument("--out", type=Path, help="Write the allocation here as CSV.")

    council = subparsers.add_parser("council", help="Expert review of a recommendation.")
    _add_common(council)
    council.add_argument("--unit", help="Unit to review. Omit to review the highest-priority units.")
    council.add_argument("--limit", type=int, default=5, help="Units to review.")
    council.add_argument(
        "--use-model",
        action="store_true",
        help="Also poll a language model for each seat. Needs ANTHROPIC_API_KEY.",
    )

    backtest = subparsers.add_parser("backtest", help="Score the reach model on held-out rounds.")
    _add_common(backtest)
    backtest.add_argument(
        "--holdout",
        type=int,
        default=2,
        help=(
            "Trailing rounds withheld. The history that remains must still hold six "
            "rounds, because the reach model cuts it three ways: rounds that fit the "
            "centre, rounds that teach the spread, and rounds that price that spread "
            "out of sample."
        ),
    )

    validate = subparsers.add_parser(
        "validate", help="Run the engine on a simulated programme with known truth."
    )
    _add_common(validate)

    args = parser.parse_args(argv)
    config = _config(args)

    if args.command == "validate":
        spec = SyntheticProgramme()
        panel, truth = generate_synthetic_panel(spec)
        plan_set = run_pipeline(
            panel, config, planning_interval_months=spec.forward_interval_months,
            require_trough=args.require_trough,
        )
        print(programme_summary(plan_set, config))
        print()
        print(validate_against_truth(plan_set, truth, max_rounds=config.max_rounds).summary())
        return 0

    panel = _load(args)

    if args.command == "backtest":
        print(summarise_backtest(
            backtest_on_panel(
                panel, config, holdout_rounds=args.holdout,
                planning_interval_months=args.interval,
            )
        ))
        return 0

    plan_set = run_pipeline(
        panel, config, planning_interval_months=args.interval,
        require_trough=args.require_trough,
    )

    if args.command == "plan":
        if args.unit:
            print(unit_brief(plan_set, args.unit, config))
            return 0
        print(programme_summary(plan_set, config))
        escalations = escalation_table(plan_set)
        if not escalations.empty:
            print()
            print("UNITS NEEDING A DIFFERENT INSTRUMENT (top 15 by gap)")
            print(escalations.head(15).to_string(index=False, float_format="%.3f"))
        if args.out:
            plan_set.to_frame().to_csv(args.out, index=False)
            print(f"\nPlan table written to {args.out}")
        return 0

    if args.command == "allocate":
        allocation = allocate_rounds(
            plan_set.plans, plan_set.parameters, config,
            round_budget=args.budget, objective=Objective(args.objective),
        )
        print(f"Allocated {allocation.rounds_allocated} of {args.budget} rounds.")
        print(
            f"Objective {allocation.objective_value:,.0f} against "
            f"{allocation.objective_uniform:,.0f} for an even spread "
            f"({allocation.uplift:+.0%})."
        )
        for note in allocation.notes:
            print(f"  {note}")
        if not allocation.table.empty:
            print()
            print(allocation.table.head(20).to_string(index=False, float_format="%.3f"))
        if args.out:
            allocation.table.to_csv(args.out, index=False)
            print(f"\nAllocation written to {args.out}")
        return 0

    if args.command == "council":
        if args.unit:
            plan = next((p for p in plan_set.plans if p.unit_id == args.unit), None)
            if plan is None:
                print(f"Unit '{args.unit}' is not in this run.", file=sys.stderr)
                return 1
            from .council import convene

            print(convene(plan, plan_set, config, use_model=args.use_model).render())
            return 0
        for record in convene_many(plan_set, config, limit=args.limit, use_model=args.use_model):
            print(record.render())
            print("\n" + "-" * 72 + "\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
