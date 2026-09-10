"""Read a panel from a delimited or Parquet file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..contracts import ValidationReport, validate_panel


def load_csv_panel(path: str | Path) -> tuple[pd.DataFrame, ValidationReport]:
    """Load a panel from CSV, TSV or Parquet and validate it against the contract.

    Args:
        path: File to read. The suffix selects the reader.

    Returns:
        A pair ``(panel, report)``.

    Raises:
        ContractError: If the file does not satisfy the contract.
        ValueError: If the suffix is not recognised.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        panel = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        panel = pd.read_csv(path)
    elif suffix in {".tsv", ".tab"}:
        panel = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .csv, .tsv or .parquet.")

    if "round_start" in panel.columns:
        panel["round_start"] = pd.to_datetime(panel["round_start"], errors="coerce")
    return panel, validate_panel(panel)
