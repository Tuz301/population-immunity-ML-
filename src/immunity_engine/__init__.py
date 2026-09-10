"""Campaign requirement engine for polio outbreak response.

Answers, per settlement, ward or LGA: how many campaign rounds are needed to
reach population immunity, how confident that number is, and when the answer is
that no number of rounds will do it.
"""

from .contracts import (
    ContractError,
    DenominatorBasis,
    EngineConfig,
    Feasibility,
    Grain,
    ProvenanceLog,
    validate_panel,
)
from .pipeline import PlanSet, run_pipeline
from .rounds_required import UnitInputs, UnitPlan, solve_unit
from .validation import ValidationResult, backtest_on_panel, validate_against_truth

__version__ = "0.1.0"

__all__ = [
    "ContractError",
    "DenominatorBasis",
    "EngineConfig",
    "Feasibility",
    "Grain",
    "PlanSet",
    "ProvenanceLog",
    "UnitInputs",
    "UnitPlan",
    "ValidationResult",
    "__version__",
    "backtest_on_panel",
    "run_pipeline",
    "solve_unit",
    "validate_against_truth",
    "validate_panel",
]
