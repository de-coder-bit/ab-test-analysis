"""
data_loader.py
==============
Load, validate, and clean the Cookie Cats A/B test dataset.

The dataset has columns:
  userid        : int   - unique player ID
  version       : str   - 'gate_30' (control) or 'gate_40' (treatment)
  sum_gamerounds: int   - total game rounds played in the first 14 days
  retention_1   : bool  - was the player back after 1 day?
  retention_7   : bool  - was the player back after 7 days?
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Expected schema constants
# ---------------------------------------------------------------------------
_EXPECTED_COLUMNS: list[str] = [
    "userid",
    "version",
    "sum_gamerounds",
    "retention_1",
    "retention_7",
]
_EXPECTED_VERSIONS: list[str] = ["gate_30", "gate_40"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_raw(path: Path) -> pd.DataFrame:
    """Load the Cookie Cats CSV and coerce all columns to their expected dtypes.

    Parameters
    ----------
    path : Path
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with enforced dtypes:
        - userid          -> int64
        - version         -> str (object)
        - sum_gamerounds  -> int64
        - retention_1     -> bool
        - retention_7     -> bool

    Raises
    ------
    FileNotFoundError
        If *path* does not exist, with a descriptive message including the
        resolved absolute path.
    ValueError
        If required columns are missing from the CSV.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Cookie Cats dataset not found.\n"
            f"  Expected location : {resolved}\n"
            f"  Please place the CSV at the path supplied to load_raw()."
        )

    df = pd.read_csv(resolved)

    # Verify required columns are present before attempting coercions
    missing_cols = [c for c in _EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing required columns: {missing_cols}.\n"
            f"  Found columns: {df.columns.tolist()}"
        )

    # Coerce dtypes ----------------------------------------------------------------
    df["userid"] = df["userid"].astype(np.int64)
    df["version"] = df["version"].astype(str).str.strip()
    df["sum_gamerounds"] = df["sum_gamerounds"].astype(np.int64)

    # Retention columns may be stored as True/False strings or 0/1 integers
    for col in ("retention_1", "retention_7"):
        series = df[col]
        if series.dtype == object:
            # Handle string representations: 'True'/'False', 'true'/'false', '1'/'0'
            df[col] = series.str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            ).astype(bool)
        else:
            df[col] = series.astype(bool)

    return df


def validate_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Inspect the dataframe for schema issues and return a summary report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (or cleaned) Cookie Cats dataframe.

    Returns
    -------
    dict[str, Any]
        A dictionary with the following keys:

        ``missing_values``
            pd.Series – count of NaN values per column.
        ``duplicate_userids``
            int – number of duplicated userid rows (rows beyond the first).
        ``version_counts``
            pd.Series – value counts of the ``version`` column.
        ``dtype_ok``
            bool – True when all columns carry their expected dtypes and only
            the two known version labels are present.
        ``expected_versions``
            list[str] – the two permitted version strings.
    """
    missing_values: pd.Series = df.isnull().sum()

    duplicate_userids: int = int(df.duplicated(subset="userid").sum())

    version_counts: pd.Series = df["version"].value_counts()

    # dtype_ok checks ----------------------------------------------------------------
    dtype_flags: list[bool] = [
        pd.api.types.is_integer_dtype(df["userid"]),
        pd.api.types.is_integer_dtype(df["sum_gamerounds"]),
        pd.api.types.is_bool_dtype(df["retention_1"]),
        pd.api.types.is_bool_dtype(df["retention_7"]),
        # version must be object/string and contain only expected labels
        df["version"].dtype == object,
        set(df["version"].unique()).issubset(set(_EXPECTED_VERSIONS)),
    ]
    dtype_ok: bool = all(dtype_flags)

    return {
        "missing_values": missing_values,
        "duplicate_userids": duplicate_userids,
        "version_counts": version_counts,
        "dtype_ok": dtype_ok,
        "expected_versions": _EXPECTED_VERSIONS,
    }


def detect_outlier(
    df: pd.DataFrame,
    col: str = "sum_gamerounds",
    threshold_pct: float = 99.9,
) -> dict[str, Any]:
    """Identify extreme outliers in *col* using a percentile threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Cookie Cats dataframe (raw or partially cleaned).
    col : str, optional
        Column to inspect for outliers. Defaults to ``'sum_gamerounds'``.
    threshold_pct : float, optional
        Percentile above which a value is considered an outlier.
        Defaults to ``99.9``.

    Returns
    -------
    dict[str, Any]
        A dictionary with the following keys:

        ``threshold_value``
            float – the value at the *threshold_pct* percentile.
        ``n_outliers``
            int – number of rows whose ``col`` value exceeds the threshold.
        ``max_value``
            float – maximum observed value in ``col``.
        ``outlier_rows``
            pd.DataFrame – subset of *df* containing outlier rows.
    """
    threshold_value: float = float(np.percentile(df[col].values, threshold_pct))
    outlier_mask: pd.Series = df[col] > threshold_value
    outlier_rows: pd.DataFrame = df[outlier_mask].copy()

    return {
        "threshold_value": threshold_value,
        "n_outliers": int(outlier_mask.sum()),
        "max_value": float(df[col].max()),
        "outlier_rows": outlier_rows,
    }


def clean(
    df: pd.DataFrame,
    remove_outliers: bool = False,
    outlier_cap_pct: float = 99.9,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove duplicates and handle outliers in ``sum_gamerounds``.

    Processing steps (in order):
    1. Drop duplicate ``userid`` rows – keep the first occurrence.
    2. Either **cap** or **drop** rows where ``sum_gamerounds`` exceeds the
       *outlier_cap_pct* percentile, depending on *remove_outliers*.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Cookie Cats dataframe produced by :func:`load_raw`.
    remove_outliers : bool, optional
        * ``False`` (default) – cap ``sum_gamerounds`` at the percentile value
          rather than removing the row entirely.
        * ``True`` – drop rows whose ``sum_gamerounds`` exceeds the threshold.
    outlier_cap_pct : float, optional
        Percentile used to define the outlier boundary. Defaults to ``99.9``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        ``(cleaned_df, report_dict)`` where *report_dict* contains:

        ``n_before``
            int – row count before cleaning.
        ``n_after``
            int – row count after cleaning.
        ``n_duplicates_removed``
            int – number of duplicate-userid rows dropped.
        ``outlier_action``
            str – ``'dropped'`` or ``'capped'``.
        ``outlier_cap_value``
            float – the percentile threshold value used.
        ``n_outliers_affected``
            int – number of rows dropped or capped.
    """
    n_before: int = len(df)

    # Step 1: Remove duplicate userids (keep first) ----------------------------------
    df_clean = df.drop_duplicates(subset="userid", keep="first").copy()
    n_duplicates_removed: int = n_before - len(df_clean)

    # Step 2: Outlier handling -------------------------------------------------------
    outlier_info = detect_outlier(df_clean, col="sum_gamerounds", threshold_pct=outlier_cap_pct)
    cap_value: float = outlier_info["threshold_value"]
    outlier_mask: pd.Series = df_clean["sum_gamerounds"] > cap_value
    n_outliers_affected: int = int(outlier_mask.sum())

    if remove_outliers:
        df_clean = df_clean[~outlier_mask].copy()
        outlier_action = "dropped"
    else:
        df_clean.loc[outlier_mask, "sum_gamerounds"] = int(cap_value)
        outlier_action = "capped"

    n_after: int = len(df_clean)

    report: dict[str, Any] = {
        "n_before": n_before,
        "n_after": n_after,
        "n_duplicates_removed": n_duplicates_removed,
        "outlier_action": outlier_action,
        "outlier_cap_value": cap_value,
        "n_outliers_affected": n_outliers_affected,
    }

    return df_clean, report


def load_and_clean(
    path: Path,
    remove_outliers: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convenience pipeline: load → validate → clean → report.

    Parameters
    ----------
    path : Path
        Path to the Cookie Cats CSV file.
    remove_outliers : bool, optional
        Passed directly to :func:`clean`. Defaults to ``False`` (cap mode).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        ``(cleaned_df, full_report)`` where *full_report* merges the
        validation report and clean report and additionally contains:

        ``zero_round_players``
            int – count of players with ``sum_gamerounds == 0`` in the
            cleaned dataframe.
        ``control_size``
            int – number of rows in the ``gate_30`` group.
        ``treatment_size``
            int – number of rows in the ``gate_40`` group.
    """
    raw_df: pd.DataFrame = load_raw(path)
    validation_report: dict[str, Any] = validate_schema(raw_df)
    cleaned_df, clean_report = clean(raw_df, remove_outliers=remove_outliers)

    # Extra summary statistics on the cleaned data -----------------------------------
    zero_round_players: int = int((cleaned_df["sum_gamerounds"] == 0).sum())
    control_size: int = int((cleaned_df["version"] == "gate_30").sum())
    treatment_size: int = int((cleaned_df["version"] == "gate_40").sum())

    full_report: dict[str, Any] = {
        **validation_report,
        **clean_report,
        "zero_round_players": zero_round_players,
        "control_size": control_size,
        "treatment_size": treatment_size,
    }

    return cleaned_df, full_report


def split_by_version(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into control and treatment groups.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Cookie Cats dataframe (output of :func:`load_and_clean` or
        :func:`clean`).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(control_df, treatment_df)`` where:

        * *control_df*   → rows where ``version == 'gate_30'``
        * *treatment_df* → rows where ``version == 'gate_40'``

        Both sub-dataframes have their index reset.
    """
    control_df: pd.DataFrame = (
        df[df["version"] == "gate_30"].copy().reset_index(drop=True)
    )
    treatment_df: pd.DataFrame = (
        df[df["version"] == "gate_40"].copy().reset_index(drop=True)
    )
    return control_df, treatment_df
