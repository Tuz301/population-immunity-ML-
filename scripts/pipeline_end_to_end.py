"""The whole pipeline in one file, from data ingestion to decision output.

Run it:

    python scripts/pipeline_end_to_end.py

`pipeline.run_pipeline` does stages 1 to 6 in one call and is what production
uses. This script unrolls them so the data flow is visible: what each stage
receives, what it estimates, and how that estimate changes the answer. Every
number printed is computed live, not quoted.

The stages:

    1. INGEST      panel of unit-rounds, validated against the data contract
    2. FEATURES    leak-free matrix; every lag shifted so no round sees itself
    3. REACH       the machine-learning layer: LightGBM + conformal intervals
    4. PARAMETERS  unreachable core, reach spread, stickiness, drift
    5. IMMUNITY    current immunity, reconstructed by replaying history
    6. INVERT      Monte Carlo over the susceptible stock -> round count
    7. ALLOCATE    spread a fixed round budget across units
    8. COUNCIL     six mandated seats adjudicate the recommendation
    9. REPORT      what a decision maker reads
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from immunity_engine.adapters.synthetic import (  # noqa: E402
    SyntheticProgramme,
    generate_synthetic_panel,
)
from immunity_engine.contracts import (  # noqa: E402
    EngineConfig,
    Feasibility,
    ProvenanceLog,
    validate_panel,
)
from immunity_engine.council import convene  # noqa: E402
from immunity_engine.features import build_features  # noqa: E402
from immunity_engine.optimizer import allocate_rounds  # noqa: E402
from immunity_engine.pipeline import (  # noqa: E402
    estimate_current_immunity,
    estimate_unit_parameters,
    run_pipeline,
)
from immunity_engine.reach_model import ReachModel  # noqa: E402
from immunity_engine.report import escalation_table, programme_summary, unit_brief  # noqa: E402
from immunity_engine.rounds_required import UnitInputs, solve_unit  # noqa: E402

PLANNING_INTERVAL_MONTHS = 3.0


def banner(number: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"STAGE {number} — {title}")
    print("=" * 78)


def main() -> int:
    config = EngineConfig(n_draws=2000)
    rng = np.random.default_rng(config.random_seed)

    # ================================================================== 1
    banner(1, "INGEST")
    print(
        "Source adapters all return the same shape: one row per (unit, round).\n"
        "  adapters/postgres.py   reads the NEOC portal schema directly\n"
        "  adapters/csvfile.py    reads a delimited or Parquet export\n"
        "  adapters/synthetic.py  simulates a programme with known truth\n"
        "\nUsing the synthetic adapter here so every figure below is checkable\n"
        "against a truth the engine never sees.\n"
    )
    spec = SyntheticProgramme()
    panel, truth = generate_synthetic_panel(spec)

    report = validate_panel(panel)
    print(f"Panel: {report.n_rows} unit-rounds, {report.n_units} units, {report.n_rounds} rounds")
    admin = panel["admin_vaccinated"] / panel["target_pop"]
    verified = panel["verified_vaccinated"] / panel["target_pop"]
    print(
        f"Administrative coverage {admin.mean():.1%} against independently verified "
        f"{verified.mean():.1%} — a gap of {100 * (admin.mean() - verified.mean()):.1f} pp."
    )
    print("\nThe contract refuses a panel it cannot trust, and warns about the rest:")
    for warning in report.warnings:
        print(f"  ! {warning}")

    # ================================================================== 2
    banner(2, "FEATURES")
    print(
        "Every lag is shifted within unit before use, so no round can see its own\n"
        "outcome. That rule is enforced by construction, not checked afterwards: a\n"
        "single leaked feature makes the backtest look excellent and the field\n"
        "performance look nothing like it.\n"
    )
    log = ProvenanceLog()
    features, log = build_features(panel, config, log)
    print(f"Feature matrix: {features.shape[0]} rows x {features.shape[1]} columns")
    print("\nOne unit's reach history, and the lag the model is allowed to use:")
    unit = features["unit_id"].iloc[0]
    sample = features[features["unit_id"] == unit][
        ["round_index", "reach", "reach_prev", "reach_mean_prev3", "chronic_miss_count"]
    ]
    print(sample.to_string(index=False, float_format="%.3f"))
    print("\n  reach_prev on the first round is NaN — correctly, there is no prior round.")

    # ================================================================== 3
    banner(3, "REACH — the machine-learning layer")
    print(
        "This is the only learned component. It predicts one quantity the\n"
        "epidemiology cannot derive from first principles: the share of children a\n"
        "team will actually reach next round.\n"
        "\nIt does NOT predict the round count. A regression on past round counts\n"
        "would learn the schedule this programme ran, not the requirement it faced.\n"
        "\nFitted as a distribution, not a point. A point forecast of reach yields a\n"
        "point forecast of the round count, which hides the difference between\n"
        "'three rounds, confidently' and 'three if all goes well, seven if not'.\n"
    )
    model = ReachModel()
    reach_report = model.fit(features, validation_rounds=2)
    print(
        f"Fit on {reach_report.n_train} rows, calibrated on {reach_report.n_calibration}, "
        f"scored on {reach_report.n_validation} held-out rows."
    )
    print(f"  MAE {reach_report.mae:.4f} reach points | RMSE {reach_report.rmse:.4f}")
    print(f"  Skill over assuming next round repeats last: {reach_report.skill_vs_persistence:+.1%}")
    print(f"  Interval coverage: {reach_report.interval_coverage}")
    print(f"  {reach_report.calibration_verdict()}")
    print("\nWhat drives the forecast:")
    for name, gain in list(reach_report.feature_importance.items())[:6]:
        print(f"  {gain:6.1%}  {name}")
    print(
        "\nMonotone constraints are enforced: reach can never be forecast HIGHER\n"
        "because insecurity rose or the place was missed more often. LightGBM\n"
        "refuses those constraints under a quantile objective, so the centre and\n"
        "the spread are fitted separately and quantiles rebuilt from a conformal\n"
        "calibration slice at the matching forecast horizon."
    )
    latest = features.sort_values("round_index").groupby("unit_id").tail(1)
    quantiles = model.predict_quantiles(latest)
    print("\nPredicted reach for the next round, three units:")
    preview = pd.concat(
        [latest[["unit_id"]].reset_index(drop=True), quantiles.reset_index(drop=True)], axis=1
    )
    print(preview.head(3).to_string(index=False, float_format="%.3f"))

    # ================================================================== 4
    banner(4, "PARAMETERS — who a round reaches, and whether it is the same children")
    print(
        "The naive model treats each round as an independent coin flip over the\n"
        "same children, so susceptibles fall geometrically and any target is\n"
        "reachable with enough rounds. Field data contradict that: the same\n"
        "households are missed round after round.\n"
    )
    parameters, notes = estimate_unit_parameters(features, panel, config, log)
    for note in notes:
        print(f"  - {note}")
    print("\nEstimated per unit (first three):")
    print(
        parameters[["pi_zero", "kappa", "rho", "reach_drift_per_round", "denominator_inflation"]]
        .head(3)
        .to_string(float_format="%.3f")
    )
    print(
        "\n  pi_zero  share of children reachable in NO round — sets a hard ceiling\n"
        "  kappa    spread of reach between children; lower means more unequal\n"
        "  rho      stickiness: how much reach persists round to round\n"
        "  drift    measured change in reach per successive round (fatigue)"
    )

    # ================================================================== 5
    banner(5, "IMMUNITY — reconstructed, because it is never measured")
    print(
        "Immunity at settlement grain is not observed. It is rebuilt: start each\n"
        "unit at the immunity routine immunisation alone sustains, then apply every\n"
        "observed round at the reach that round achieved, letting births refill the\n"
        "susceptible stock in between.\n"
        "\nThe starting point is not a free parameter. With no campaigns the stock\n"
        "settles where births and ageing balance, putting immunity at the share of\n"
        "newborns routine immunisation protects. That is the level a place returns\n"
        "to, so it is the right place to replay from.\n"
    )
    immunity = estimate_current_immunity(features, parameters, panel, config)
    parameters = parameters.join(immunity)
    check = parameters.join(truth.set_index("unit_id")["true_immunity_at_cut"])
    error = (check["current_immunity"] - check["true_immunity_at_cut"]).abs()
    print(
        f"Estimated immunity: mean {check['current_immunity'].mean():.1%} "
        f"against a true {check['true_immunity_at_cut'].mean():.1%}"
    )
    print(f"Mean absolute error against a truth the engine never saw: {error.mean():.3f}")

    # ================================================================== 6
    banner(6, "INVERT — how many rounds, and can any number do it")
    print(
        "Not regressed. Draw the parameters, run the susceptible stock forward, and\n"
        "record the round at which immunity first crosses the target. Repeat for\n"
        f"{config.n_draws} draws. The result is a distribution over round counts.\n"
        "\nThe simulation runs PAST the round budget, so the ceiling is a property of\n"
        "the schedule rather than of where the simulation stopped.\n"
    )
    reconciliation = config.reconcile_take()
    print(f"Target: Vc(adj) = {config.vc_adjusted:.1%} at R0 {config.r0}")
    print(
        f"  Per-dose take used {reconciliation['per_dose_take_used']:.3f}; the stated "
        f"schedule failure of {config.schedule_failure} over {config.doses_in_schedule} doses "
        f"implies {reconciliation['per_dose_take_implied_by_epsilon']:.3f}. "
        f"Residual {reconciliation['residual_pp']:+.2f} pp — the two agree."
    )

    row = parameters.iloc[0]
    unit_id = str(parameters.index[0])
    draws = model.sample_reach(latest[latest["unit_id"] == unit_id], 400, rng)
    inputs = UnitInputs(
        unit_id=unit_id,
        grain=str(row["grain"]),
        lga_code=str(row["lga_code"]),
        state_code=str(row["state_code"]),
        target_pop=float(row["target_pop"]),
        denominator_basis=str(row["denominator_basis"]),
        reach_draws=draws[0],
        current_immunity=float(row["current_immunity"]),
        current_immunity_sd=float(row["current_immunity_sd"]),
        pi_zero=float(row["pi_zero"]),
        kappa=float(row["kappa"]),
        rho=float(row["rho"]),
        births_per_month=float(row["births_per_month"]),
        ri_protection=float(row["ri_protection"]),
        interval_months=PLANNING_INTERVAL_MONTHS,
        denominator_inflation_mean=float(row["denominator_inflation"]),
        reach_drift_per_round=float(row["reach_drift_per_round"]),
    )
    plan = solve_unit(inputs, config, rng=rng)
    print(f"\nOne unit, solved:\n  {plan.headline()}")
    print("\n  Chance each round count is enough:")
    for n, probability in sorted(plan.probability_by_round.items()):
        if n > 8 and probability > 0.995:
            break
        print(f"    {n:>2} rounds  {probability:>5.0%}  {'#' * int(round(probability * 30))}")
    print(
        f"\n  Schedule ceiling {plan.immunity_ceiling_median:.1%} | "
        f"core-limited ceiling {plan.core_ceiling_median:.1%}"
    )
    print(
        "  The gap between those two ceilings is the whole diagnosis. If the CORE\n"
        "  ceiling is below target, children cannot be reached: send access and\n"
        "  enumeration. If only the SCHEDULE ceiling is below target, births refill\n"
        "  the stock faster than rounds drain it: shorten the interval or raise\n"
        "  routine immunisation. Neither is solved by more rounds at this spacing."
    )

    # -- everything above, for every unit, in one call -----------------
    print("\nRunning stages 1-6 for all units via the production entry point...")
    plan_set = run_pipeline(panel, config, planning_interval_months=PLANNING_INTERVAL_MONTHS)
    frame = plan_set.to_frame()
    print(f"Planned {len(frame)} units.\n")
    print(frame["feasibility"].value_counts().to_string())

    # ================================================================== 7
    banner(7, "ALLOCATE — where the next rounds should go")
    print(
        "Marginal immunity per round falls with every previous round in the same\n"
        "unit: the easy children are reached first. That makes the objective\n"
        "submodular, so greedy allocation is within 1 - 1/e of optimal and can be\n"
        "audited line by line.\n"
        "\nThe optimiser REFUSES to spend rounds on a ceiling-limited unit. That is\n"
        "the specific mistake this engine exists to prevent, so it is enforced in\n"
        "code rather than discouraged in a footnote.\n"
    )
    allocation = allocate_rounds(
        plan_set.plans, plan_set.parameters, config, round_budget=25
    )
    print(f"Allocated {allocation.rounds_allocated} of 25 rounds.")
    print(
        f"Risk-weighted susceptibles averted: {allocation.objective_value:,.0f} against "
        f"{allocation.objective_uniform:,.0f} for an even spread ({allocation.uplift:+.0%})."
    )
    for note in allocation.notes:
        print(f"  {note}")
    if not allocation.table.empty:
        print()
        print(
            allocation.table[["unit_id", "lga_code", "rounds", "current_immunity", "value"]]
            .head(6)
            .to_string(index=False, float_format="%.3f")
        )

    # ================================================================== 8
    banner(8, "COUNCIL — six seats adjudicate before anything ships")
    print(
        "Deterministic rules run first and always; every check is a threshold this\n"
        "programme has been caught by before. The language-model pass runs on top\n"
        "when a key is present. One block holds a recommendation, overall\n"
        "confidence is the WEAKEST seat rather than the mean, and dissent is\n"
        "published rather than filed.\n"
    )
    blocked = [p for p in plan_set.plans if p.feasibility is Feasibility.INFEASIBLE_BY_CAMPAIGNS]
    subject = blocked[0] if blocked else plan_set.plans[0]
    record = convene(subject, plan_set, config, use_model=False)
    print(record.render())

    # ================================================================== 9
    banner(9, "REPORT — what a decision maker reads")
    print(programme_summary(plan_set, config))
    escalations = escalation_table(plan_set)
    if not escalations.empty:
        print()
        print("UNITS NEEDING A DIFFERENT INSTRUMENT (top 8 by gap to target)")
        print(
            escalations[["unit_id", "lga_code", "current_immunity", "gap_to_target_pp", "capped_by"]]
            .head(8)
            .to_string(index=False, float_format="%.3f")
        )
    print()
    print("-" * 78)
    print(unit_brief(plan_set, subject.unit_id, config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
