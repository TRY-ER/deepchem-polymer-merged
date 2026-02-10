"""Utility to produce a detailed textual analysis of a column in a pandas DataFrame.

Function: analyze_column_as_text(df, column, ...)

Returns a multiline string summary that covers:
- general counts and nulls
- dtype & uniqueness
- value frequency (top-N)
- numeric statistics and outlier counts (if numeric)
- text-specific statistics (lengths, tokens) for object dtypes
- datetime-specific stats for datetime dtypes

Example usage:
    from analyze_sa_score import analyze_column_as_text
    print(analyze_column_as_text(df, 'column_name', top_n=10))

"""
from typing import Any, Optional
import pandas as pd
import numpy as np


def _pct(part: int, total: int, digits: int = 2) -> str:
    if total == 0:
        return "0%"
    return f"{100.0 * part / total:.{digits}f}%"


def _safe_len(x: Any) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def analyze_column_as_text(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
    sample_size: int = 5,
    round_digits: int = 4,
    separator: str = ">>",
) -> str:
    """Return a detailed textual analysis of a single column in a DataFrame.

    This version supports columns that contain either float values or long strings where
    the meaningful value appears before a separator (default '>>'). The function will
    extract the prefix before the separator for textual analysis and coerce numeric-looking
    prefixes to floats for numeric statistics.

    Parameters
    - df: pandas DataFrame
    - column: column name to analyze
    - top_n: how many top values to include
    - sample_size: how many example values to show for rare/unique samples
    - round_digits: digits to round numeric summaries
    - separator: string separator used to split long string entries and retain the prefix

    Returns
    - str: multi-line text report
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    s = df[column]
    total = len(s)
    non_null = s.notna().sum()
    null_count = total - non_null

    parts = []
    parts.append(f"Column: {column}")
    parts.append(f"Total rows: {total}")
    parts.append(f"Non-null: {non_null} ({_pct(non_null, total)})")
    parts.append(f"Nulls: {null_count} ({_pct(null_count, total)})")
    parts.append(f"Dtype: {s.dtype}")

    # Build `primary` values by taking prefix before the separator for strings
    def _extract_primary(x):
        if pd.isna(x):
            return np.nan
        # If already a number, keep it
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return x
        # convert bytes to str
        if isinstance(x, bytes):
            try:
                x = x.decode("utf-8", errors="ignore")
            except Exception:
                x = str(x)
        if isinstance(x, str):
            v = x.strip()
            if separator and separator in v:
                v = v.split(separator, 1)[0].strip()
            return v
        return str(x)

    primary = s.map(_extract_primary)

    # Updated uniqueness and top-values based on primary values
    try:
        unique_count = primary.nunique(dropna=True)
    except Exception:
        unique_count = len(pd.unique(primary.dropna()))
    parts.append(f"Unique (non-null) primary values: {unique_count} ({_pct(unique_count, non_null)})")

    # value counts on the cleaned/primary values
    try:
        vc_primary = primary.dropna().astype(str).value_counts()
    except Exception:
        vc_primary = pd.Series([], dtype=int)

    if not vc_primary.empty:
        parts.append("")
        parts.append(f"Top {min(top_n, len(vc_primary))} primary values (prefix before '{separator}'):")
        for i, (val, cnt) in enumerate(vc_primary.head(top_n).items(), start=1):
            parts.append(f"  {i}. {repr(val)} — {cnt} ({_pct(cnt, non_null)})")

    # Now separate numeric-like vs text-like among primary values
    numeric_coerced = pd.to_numeric(primary, errors="coerce")
    num_non_na = numeric_coerced.notna().sum()
    text_non_na = primary.dropna().shape[0] - num_non_na
    parts.append("")
    parts.append(f"Primary types: numeric-like: {num_non_na} ({_pct(num_non_na, non_null)}), text-like: {text_non_na} ({_pct(text_non_na, non_null)})")

    # Simplified handling: only numeric-like and text-like primary values are analysed.
    # Numeric summary when numeric-like primary values exist
    if num_non_na > 0:
        arr = numeric_coerced.dropna().astype(float)
        q1 = arr.quantile(0.25)
        q3 = arr.quantile(0.75)
        iqr = q3 - q1
        mean = arr.mean()
        median = arr.median()
        std = arr.std()
        mn = arr.min()
        mx = arr.max()
        skew = arr.skew()
        kurt = arr.kurt()
        zeros = (arr == 0).sum()
        negatives = (arr < 0).sum()
        positives = (arr > 0).sum()
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((arr < lower_bound) | (arr > upper_bound)).sum()

        parts.append("")
        parts.append("Numeric summary (from primary values):")
        parts.append(f"  Count: {len(arr)}")
        parts.append(f"  Mean: {mean:.{round_digits}f}")
        parts.append(f"  Median: {median:.{round_digits}f}")
        parts.append(f"  Std: {std:.{round_digits}f}")
        parts.append(f"  Min: {mn:.{round_digits}f}")
        parts.append(f"  25%: {q1:.{round_digits}f}")
        parts.append(f"  75%: {q3:.{round_digits}f}")
        parts.append(f"  Max: {mx:.{round_digits}f}")
        parts.append(f"  Skewness: {skew:.{round_digits}f}")
        parts.append(f"  Kurtosis: {kurt:.{round_digits}f}")
        parts.append(f"  Zeros: {zeros} ({_pct(zeros, len(arr))})")
        parts.append(f"  Negatives: {negatives} ({_pct(negatives, len(arr))})")
        parts.append(f"  Positives: {positives} ({_pct(positives, len(arr))})")
        parts.append(f"  Outliers (1.5*IQR rule): {outliers} ({_pct(outliers, len(arr))})")

        parts.append("")
        parts.append("  Smallest 5 values:")
        for v in arr.nsmallest(5):
            parts.append(f"    - {v:.{round_digits}f}")
        parts.append("  Largest 5 values:")
        for v in arr.nlargest(5):
            parts.append(f"    - {v:.{round_digits}f}")

    # Textual summary when text-like primary values exist
    if text_non_na > 0:
        txt = primary[pd.isna(numeric_coerced)].dropna().astype(str)
        lengths = txt.map(_safe_len)
        mean_len = lengths.mean()
        median_len = lengths.median()
        min_len = lengths.min()
        max_len = lengths.max()

        numeric_like_count = txt.str.fullmatch(r"[-+]?\d*(?:\.\d+)?(?:[eE][-+]?\d+)?").sum()
        has_whitespace = txt.str.contains(r"\s").sum()

        parts.append("")
        parts.append("Text/categorical summary (primary prefixes):")
        parts.append(f"  Count: {len(txt)}")
        parts.append(f"  Shortest length: {min_len}")
        parts.append(f"  Longest length: {max_len}")
        parts.append(f"  Mean length: {mean_len:.{round_digits}f}")
        parts.append(f"  Median length: {median_len}")
        parts.append(f"  Numeric-like entries among text: {numeric_like_count} ({_pct(numeric_like_count, len(txt))})")
        parts.append(f"  Entries containing whitespace: {has_whitespace} ({_pct(has_whitespace, len(txt))})")

        parts.append("")
        parts.append(f"Top {min(top_n, len(vc_primary))} frequent primary values:")
        for i, (val, cnt) in enumerate(vc_primary.head(top_n).items(), start=1):
            parts.append(f"  {i}. {repr(val)} — {cnt} ({_pct(cnt, non_null)})")

        if unique_count > top_n:
            rare = vc_primary.tail(sample_size).index.tolist()
            parts.append("")
            parts.append(f"Sample of {len(rare)} rare primary values:")
            for v in rare:
                parts.append(f"  - {repr(v)}")

    if num_non_na == 0 and text_non_na == 0:
        parts.append("")
        parts.append("No analyzable primary values found (after splitting and coercion).")

    return "\n".join(parts)


if __name__ == "__main__":
    # Focused demo for float-like values and '>>' prefixed long strings
    demo = pd.DataFrame(
        {
            # float-only column (numeric stats)
            "float_only": [1.0, 2.0, 2.0, 3.0, 100.0, np.nan, -1.0, 0.0, 2.0],
            # long strings where meaningful value is before '>>'
            "long_str": ["Alpha>>code1", "Alpha>>code2", "Beta>>x", "Gamma>>y", None, "", "Zeta>>123", "Gamma", "Delta>>ignored"],
            # mixed column containing numeric values, numeric strings, and '>>' items
            "mixed": ["Alpha>>code1", "Alpha>>code2", "3.14", 2.0, "Beta>>x", None, "42", "Gamma", "42>>ignored"],
        }
    )

    # Print focused analyses
    print(analyze_column_as_text(demo, "float_only"))
    print("\n---\n")
    print(analyze_column_as_text(demo, "long_str"))
    print("\n---\n")
    print(analyze_column_as_text(demo, "mixed"))
