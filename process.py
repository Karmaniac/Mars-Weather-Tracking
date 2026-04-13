"""
2_process.py
------------
Processes all downloaded InSight TWINS and PS CSVs into a single merged
Parquet file of per-minute averages.

Strategy:
  - Process one chunk folder at a time to control peak RAM usage
  - Minute-average each chunk immediately after loading, then free raw data
  - Accumulate the small averaged results, then merge and save at the end

Output:
    data/processed/insight_merged.parquet

Columns in output:
    sol, hour, minute, LMST_minute          ← time index
    PRESSURE, PRESSURE_TEMP                 ← from PS
    BMY_HORIZONTAL_WIND_SPEED               ← from TWINS
    BMY_VERTICAL_WIND_SPEED
    BMY_AIR_TEMP
    BMY_AVE_ROD_TEMP
    BPY_HORIZONTAL_WIND_SPEED
    BPY_VERTICAL_WIND_SPEED
    BPY_AIR_TEMP
    BPY_AVE_ROD_TEMP
    BMY_AIR_CONF, BPY_AIR_CONF             ← derived confidence scores
    BMY_WIND_CONF, BPY_WIND_CONF
"""

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ── Config ────────────────────────────────────────────────────────────────────

RAW_ROOT  = Path("data/raw")
OUT_DIR   = Path("data/processed")
OUT_FILE  = OUT_DIR / "insight_merged.parquet"

TWINS_ROOT = RAW_ROOT / "twins"
PS_ROOT    = RAW_ROOT / "ps"

# Columns to keep from PS before averaging
PS_KEEP = ["LMST", "PRESSURE", "PRESSURE_TEMP"]

# Columns to keep from TWINS before averaging
TWINS_KEEP = [
    "LMST",
    "BMY_HORIZONTAL_WIND_SPEED", "BMY_VERTICAL_WIND_SPEED", "BMY_AIR_TEMP",
    "BMY_BASE_ROD_TEMP", "BMY_MID_ROD_TEMP", "BMY_TIP_ROD_TEMP",
    "BPY_HORIZONTAL_WIND_SPEED", "BPY_VERTICAL_WIND_SPEED", "BPY_AIR_TEMP",
    "BPY_BASE_ROD_TEMP", "BPY_MID_ROD_TEMP", "BPY_TIP_ROD_TEMP",
    "BMY_AIR_TEMP_OPERATIONAL_FLAGS", "BPY_AIR_TEMP_OPERATIONAL_FLAGS",
    "BMY_WS_OPERATIONAL_FLAGS",       "BPY_WS_OPERATIONAL_FLAGS",
]

# ── Confidence helpers (from your original notebook) ─────────────────────────

def _parse_bitstr(val, expected_len: int) -> float | None:
    if pd.isna(val):
        return np.nan
    bitstr = str(val).strip()
    if bitstr.endswith(".0"):
        bitstr = bitstr[:-2]
    if len(bitstr) != expected_len or not set(bitstr).issubset({"0", "1"}):
        return np.nan
    return sum(int(b) for b in bitstr) / expected_len

def air_temp_confidence(val):
    return _parse_bitstr(val, 4)

def wind_confidence(val):
    return _parse_bitstr(val, 3)

# ── LMST parsing ──────────────────────────────────────────────────────────────

def parse_lmst(df: pd.DataFrame) -> pd.DataFrame:
    """Extract sol, hour, minute from LMST column (format: 00123M14:05:32)."""
    sol_extracted  = df["LMST"].str.extract(r"(\d+)M")
    time_part      = df["LMST"].str.extract(r"M(\d+):(\d+):")

    df["sol"]    = sol_extracted[0]
    df["hour"]   = time_part[0]
    df["minute"] = time_part[1]

    # Drop rows where LMST didn't match the expected format
    df = df.dropna(subset=["sol", "hour", "minute"])

    df["sol"]    = df["sol"].astype(int)
    df["hour"]   = df["hour"].astype(int)
    df["minute"] = df["minute"].astype(int)

    return df

def make_lmst_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a human-readable LMST string at minute resolution."""
    df["LMST_minute"] = (
        df["sol"].astype(str).str.zfill(5) + "M" +
        df["hour"].astype(str).str.zfill(2) + ":" +
        df["minute"].astype(str).str.zfill(2)
    )
    return df

# ── Per-chunk loaders ─────────────────────────────────────────────────────────

def load_csvs(folder: Path, keep_cols: list[str]) -> pd.DataFrame | None:
    """
    Load all CSVs in a folder, keeping only the columns we need.
    Returns None if no CSVs found or all fail.
    """
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        return None

    frames = []
    for f in csv_files:
        try:
            # Read with comment='#' to skip PDS label lines if present
            df = pd.read_csv(f, comment="#", low_memory=False)

            # Normalize column names (strip whitespace)
            df.columns = df.columns.str.strip()

            # Keep only columns that exist in this file
            cols = [c for c in keep_cols if c in df.columns]
            if "LMST" not in cols:
                continue  # can't do anything without time
            frames.append(df[cols])
        except Exception as e:
            print(f"    [warn] Could not read {f.name}: {e}")

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def process_ps_chunk(folder: Path) -> pd.DataFrame | None:
    """Load PS chunk → minute average."""
    df = load_csvs(folder, PS_KEEP)
    if df is None or df.empty:
        return None

    df = parse_lmst(df)
    df = df.dropna(subset=["sol", "hour", "minute"])

    result = (
        df.groupby(["sol", "hour", "minute"], sort=True)
        .agg(PRESSURE=("PRESSURE", "mean"),
             PRESSURE_TEMP=("PRESSURE_TEMP", "mean"))
        .reset_index()
    )
    return result


def process_twins_chunk(folder: Path) -> pd.DataFrame | None:
    """Load TWINS chunk → derive features → minute average."""
    df = load_csvs(folder, TWINS_KEEP)
    if df is None or df.empty:
        return None

    df = parse_lmst(df)
    df = df.dropna(subset=["sol", "hour", "minute"])

    # Average rod temps into a single value per boom
    for boom in ["BMY", "BPY"]:
        rod_cols = [f"{boom}_BASE_ROD_TEMP", f"{boom}_MID_ROD_TEMP", f"{boom}_TIP_ROD_TEMP"]
        existing = [c for c in rod_cols if c in df.columns]
        if existing:
            df[f"{boom}_AVE_ROD_TEMP"] = df[existing].mean(axis=1)

    # Derive confidence scores from operational flag bitstrings
    for boom in ["BMY", "BPY"]:
        air_col  = f"{boom}_AIR_TEMP_OPERATIONAL_FLAGS"
        wind_col = f"{boom}_WS_OPERATIONAL_FLAGS"
        if air_col in df.columns:
            df[f"{boom}_AIR_CONF"] = df[air_col].apply(air_temp_confidence)
        if wind_col in df.columns:
            df[f"{boom}_WIND_CONF"] = df[wind_col].apply(wind_confidence)

    # Columns to average at minute level
    avg_cols = [
        c for c in [
            "BMY_HORIZONTAL_WIND_SPEED", "BMY_VERTICAL_WIND_SPEED",
            "BMY_AIR_TEMP",              "BMY_AVE_ROD_TEMP",
            "BMY_AIR_CONF",              "BMY_WIND_CONF",
            "BPY_HORIZONTAL_WIND_SPEED", "BPY_VERTICAL_WIND_SPEED",
            "BPY_AIR_TEMP",              "BPY_AVE_ROD_TEMP",
            "BPY_AIR_CONF",              "BPY_WIND_CONF",
        ]
        if c in df.columns
    ]

    result = (
        df.groupby(["sol", "hour", "minute"], sort=True)[avg_cols]
        .mean()
        .reset_index()
    )
    return result

# ── Main pipeline ─────────────────────────────────────────────────────────────

def get_chunk_folders(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    twins_chunks = get_chunk_folders(TWINS_ROOT)
    ps_chunks    = get_chunk_folders(PS_ROOT)

    if not twins_chunks:
        print(f"[error] No chunk folders found in {TWINS_ROOT}")
        return
    if not ps_chunks:
        print(f"[error] No chunk folders found in {PS_ROOT}")
        return

    print(f"Found {len(twins_chunks)} TWINS chunks, {len(ps_chunks)} PS chunks.\n")

    # ── Process TWINS chunks ──────────────────────────────────────────────────
    print("=== Processing TWINS ===")
    twins_averaged = []
    for folder in twins_chunks:
        print(f"  {folder.name} ...", end=" ", flush=True)
        result = process_twins_chunk(folder)
        if result is not None and not result.empty:
            twins_averaged.append(result)
            print(f"{len(result):,} minute-rows")
        else:
            print("no data")
        gc.collect()

    # ── Process PS chunks ─────────────────────────────────────────────────────
    print("\n=== Processing PS ===")
    ps_averaged = []
    for folder in ps_chunks:
        print(f"  {folder.name} ...", end=" ", flush=True)
        result = process_ps_chunk(folder)
        if result is not None and not result.empty:
            ps_averaged.append(result)
            print(f"{len(result):,} minute-rows")
        else:
            print("no data")
        gc.collect()

    # ── Combine and merge ─────────────────────────────────────────────────────
    print("\n=== Combining and merging ===")

    if not twins_averaged:
        print("[error] No TWINS data processed.")
        return
    if not ps_averaged:
        print("[error] No PS data processed.")
        return

    twins_df = pd.concat(twins_averaged, ignore_index=True)
    ps_df    = pd.concat(ps_averaged,    ignore_index=True)

    del twins_averaged, ps_averaged
    gc.collect()

    print(f"TWINS averaged shape: {twins_df.shape}")
    print(f"PS averaged shape:    {ps_df.shape}")

    merged = pd.merge(
        twins_df, ps_df,
        on=["sol", "hour", "minute"],
        how="outer"       # keep rows even if one instrument has gaps
    )

    del twins_df, ps_df
    gc.collect()

    # Sort chronologically
    merged = merged.sort_values(["sol", "hour", "minute"]).reset_index(drop=True)

    # Add readable LMST string
    merged = make_lmst_minute(merged)

    print(f"Merged shape: {merged.shape}")
    print(f"Sol range:    {merged['sol'].min()} – {merged['sol'].max()}")
    print(f"Columns:      {list(merged.columns)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    merged.to_parquet(OUT_FILE, index=False, engine="pyarrow", compression="snappy")
    print(f"\nSaved to {OUT_FILE.resolve()}")
    print(f"File size: {OUT_FILE.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
