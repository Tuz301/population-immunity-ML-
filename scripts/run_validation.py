"""Validate the engine end to end on a simulated programme with known truth.

Run from the repository root:

    python scripts/run_validation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np  # noqa: E402

from immunity_engine.adapters.synthetic import (  # noqa: E402
    SyntheticProgramme,
    generate_synthetic_panel,
)
from immunity_engine.contracts import EngineConfig  # noqa: E402
from immunity_engine.pipeline import run_pipeline  # noqa: E402
from immunity_engine.validation import (  # noqa: E402
    backtest_on_panel,
    summarise_backtest,
    validate_against_truth,
)


def main() -> int:
    spec = SyntheticProgramme()
    panel, truth = generate_synthetic_panel(spec)
    config = EngineConfig(n_draws=2000)

    print("=" * 78)
    print("SIMULATED PROGRAMME")
    print("=" * 78)
    admin = panel["admin_vaccinated"] / panel["target_pop"]
    verified = panel["verified_vaccinated"] / panel["target_pop"]
    print(
        f"{panel['unit_id'].nunique()} settlements, {panel['round_index'].nunique()} reported rounds, "
        f"{len(panel)} unit-rounds"
    )
    print(
        f"Administrative coverage {admin.mean():.1%} against verified {verified.mean():.1%}, "
        f"a gap of {100 * (admin.mean() - verified.mean()):.1f} pp"
    )
    finite = np.isfinite(truth["true_rounds_needed"])
    print(
        f"True answer: {finite.sum()} settlements reach the target within "
        f"{spec.forward_rounds} further rounds, {(~finite).sum()} never do"
    )

    print()
    print("=" * 78)
    print("ENGINE RUN")
    print("=" * 78)
    plan_set = run_pipeline(
        panel, config, planning_interval_months=spec.forward_interval_months
    )
    report = plan_set.reach_report
    print(
        f"Reach model: MAE {report.mae:.4f}, RMSE {report.rmse:.4f}, "
        f"skill over persistence {report.skill_vs_persistence:+.1%}"
    )
    print(f"Reach interval coverage: {report.interval_coverage}")
    print(report.calibration_verdict())
    print("Top reach predictors:")
    for name, gain in list(report.feature_importance.items())[:6]:
        print(f"  {gain:6.1%}  {name}")
    print()
    for note in plan_set.notes:
        print(f"  - {note}")

    print()
    print("=" * 78)
    print("VALIDATION AGAINST KNOWN TRUTH")
    print("=" * 78)
    result = validate_against_truth(plan_set, truth, max_rounds=config.max_rounds)
    print(result.summary())
    print()
    print("Reliability of 'N rounds will be enough':")
    print(f"  {'predicted':>10}  {'observed':>9}  {'n':>6}")
    for point in result.reliability:
        print(f"  {point.predicted:>10.2f}  {point.observed:>9.2f}  {point.n:>6}")

    print()
    print("=" * 78)
    print("BACKTEST ON THE PANEL ALONE (no simulated truth used)")
    print("=" * 78)
    scored = backtest_on_panel(
        panel, config, holdout_rounds=2, planning_interval_months=spec.forward_interval_months
    )
    print(summarise_backtest(scored))

    print()
    print("=" * 78)
    print("WHERE MORE ROUNDS IS THE WRONG INSTRUMENT")
    print("=" * 78)
    escalations = plan_set.escalations()
    print(f"{len(escalations)} of {len(plan_set.plans)} settlements are ceiling-limited.")
    if not escalations.empty:
        columns = [
            "unit_id", "lga_code", "current_immunity", "immunity_ceiling_median",
            "core_ceiling_median", "probability_infeasible",
        ]
        print(escalations[columns].head(8).to_string(index=False, float_format="%.3f"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
