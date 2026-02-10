"""Utilities to filter DataFrames by SA score and export results.

Functions:
- filter_sa_score(df, column='SA_score', threshold=4.0, output_csv_path=None, drop_non_numeric=True)
- filter_sa_score_from_csv(input_csv, output_csv, column='SA_score', threshold=4.0)

Usage:
    filtered = filter_sa_score(df, column='SA_score', threshold=4.0, output_csv_path='out.csv')

"""
from typing import Optional
import pandas as pd
import numpy as np


def filter_sa_score(
    df: pd.DataFrame,
    column: str = "SA_score",
    threshold: float = 4.0,
    output_csv_path: Optional[str] = None,
    drop_non_numeric: bool = True,
) -> pd.DataFrame:
    """Return rows where the given SA score column is < threshold and optionally save to CSV.

    - Coerces the column to numeric, treating non-numeric values as NaN.
    - By default, non-numeric (NaN) rows are dropped before filtering. Set
      `drop_non_numeric=False` to keep rows with non-numeric values (they will not pass < threshold test).

    Args:
        df: input DataFrame
        column: column name containing SA scores
        threshold: numeric threshold (rows with score < threshold are kept)
        output_csv_path: if provided, writes filtered DataFrame to this CSV path
        drop_non_numeric: whether to drop rows where column can't be coerced to numeric

    Returns:
        Filtered pandas DataFrame (not including rows with NaN in the coerced score when drop_non_numeric=True)
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    # Make a copy to avoid mutating caller's DataFrame
    work = df.copy()

    # Coerce to numeric; preserve original column under a temporary name
    coerced = pd.to_numeric(work[column], errors="coerce")
    work = work.assign(_sa_coerced=coerced)

    if drop_non_numeric:
        before = len(work)
        work = work[work["_sa_coerced"].notna()]
        after = len(work)
        # optional: the caller can check counts

    filtered = work[work["_sa_coerced"] < float(threshold)].drop(columns=["_sa_coerced"])

    if output_csv_path:
        # Write out to CSV
        filtered.to_csv(output_csv_path, index=False)

    return filtered


def filter_sa_score_from_csv(
    input_csv: str,
    output_csv: str,
    column: str = "SA_score",
    threshold: float = 4.0,
) -> pd.DataFrame:
    """Read CSV, filter rows with SA score < threshold, and write to output CSV.

    Returns the filtered DataFrame after writing it.
    """
    df = pd.read_csv(input_csv)
    filtered = filter_sa_score(df, column=column, threshold=threshold, output_csv_path=output_csv)
    return filtered


if __name__ == "__main__":
    # Demo usage
    demo = pd.DataFrame(
        {
            "SA_score": [3.5, 4.2, "2.9", "not_a_number", 0.5, 5, 3],
            "mol": ["A", "B", "C", "D", "E", "F", "G"],
        }
    )
    print("Original DataFrame:\n", demo)
    filtered = filter_sa_score(demo, column="SA_score", threshold=4.0, output_csv_path="sa_filtered_demo.csv")
    print("\nFiltered rows (SA_score < 4):\n", filtered)
    print("\nSaved filtered results to 'sa_filtered_demo.csv' (in current working directory).")
