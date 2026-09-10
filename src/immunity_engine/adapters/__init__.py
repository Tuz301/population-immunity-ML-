"""Sources the engine can read a panel from."""

from .csvfile import load_csv_panel
from .postgres import NEOC_PANEL_SQL, load_postgres_panel
from .synthetic import SyntheticProgramme, generate_synthetic_panel

__all__ = [
    "NEOC_PANEL_SQL",
    "SyntheticProgramme",
    "generate_synthetic_panel",
    "load_csv_panel",
    "load_postgres_panel",
]
