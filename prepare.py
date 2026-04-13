"""
3_prepare.py
------------
Explores the merged parquet, reports data quality, then builds a normalized
sliding-window dataset ready for LSTM training.

Outputs:
    data/processed/insight_scaled.parquet   ← normalized, gap-filled data
    data/processed/scaler_params.json       ← mean/std per column for inverse transform
    data/processed/splits.json             ← sol boundaries for train/val/test
    data/processed/X_train.npy, y_train.npy
    data/processed/X_val.npy,   y_val.npy
    data/processed/X_test.npy,  y_test.npy
"""

import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

PARQUET_FILE = Path("data/processed/insight_merged.parquet")
OUT_DIR      = Path("data/processed")

FEATURE_COLS = [
    "PRESSURE",
    "BMY_AVE_ROD_TEMP",
    "BMY_HORIZONTAL_WIND_SPEED",
    "PRESSURE_TEMP",
]

TARGET_COLS = ["PRESSURE", "BMY_AVE_ROD_TEMP"]

# Sliding window: how many minutes of history to use as input
WINDOW_SIZE = 120   # 2 hours of context

# How far ahead to predict
FORECAST_HORIZON = 60   # 1 hour ahead

# Max consecutive NaN minutes to forward-fill (gaps longer than this are left as NaN)
MAX_FILL_GAP = 10

# Train / val / test split as fractions of total sols
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15  (implicit)

# ── Step 1: Load and report ───────────────────────────────────────────────────

def load_and_report(path: Path) -> pd.DataFrame:
    print("=== Loading parquet ===")
    df = pd.read_parquet(path)
    print(f"Shape:      {df.shape}")
    print(f"Sol range:  {df['sol'].min()} – {df['sol'].max()}")
    print(f"Columns:    {list(df.columns)}\n")

    print("=== NaN report ===")
    for col in FEATURE_COLS + [c for c in TARGET_COLS if c not in FEATURE_COLS]:
        if col in df.columns:
            n_nan  = df[col].isna().sum()
            pct    = 100 * n_nan / len(df)
            print(f"  {col:<40} {n_nan:>8,} NaN  ({pct:.1f}%)")
        else:
            print(f"  {col:<40}  *** COLUMN NOT FOUND ***")

    print()
    return df

# ── Step 2: Gap analysis ──────────────────────────────────────────────────────

def gap_report(df: pd.DataFrame):
    """Report on large continuous NaN stretches in the target columns."""
    print("=== Gap analysis (consecutive NaN runs > 60 min) ===")
    for col in TARGET_COLS:
        if col not in df.columns:
            continue
        is_nan = df[col].isna()
        # Label consecutive NaN runs
        run_id   = (is_nan != is_nan.shift()).cumsum()
        runs     = df[is_nan].groupby(run_id).size()
        big_runs = runs[runs > 60]
        if big_runs.empty:
            print(f"  {col}: no gaps > 60 min")
        else:
            print(f"  {col}: {len(big_runs)} gaps > 60 min  "
                  f"(longest: {big_runs.max()} min)")
    print()

# ── Step 3: Clean and sort ────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("=== Cleaning ===")

    # Keep only the columns we need
    keep = ["sol", "hour", "minute", "LMST_minute"] + list(
        dict.fromkeys(FEATURE_COLS + TARGET_COLS)  # dedup, preserve order
    )
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Sort chronologically
    df = df.sort_values(["sol", "hour", "minute"]).reset_index(drop=True)

    # Forward-fill short gaps only
    for col in FEATURE_COLS + TARGET_COLS:
        if col not in df.columns:
            continue
        before = df[col].isna().sum()
        df[col] = df[col].ffill(limit=MAX_FILL_GAP)
        after  = df[col].isna().sum()
        filled = before - after
        if filled > 0:
            print(f"  Forward-filled {filled:,} NaNs in {col} (limit={MAX_FILL_GAP})")

    # Drop rows where any target is still NaN (long gaps, unfillable)
    before = len(df)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with unfillable target NaNs")

    print(f"  Clean shape: {df.shape}\n")
    return df

# ── Step 4: Cyclical time encoding ───────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode hour and minute as sin/cos pairs so the model sees
    time-of-sol as a continuous cycle, not a discontinuous integer.
    A Martian sol has 1440 minutes (same as Earth by convention in LMST).
    """
    minute_of_sol = df["hour"] * 60 + df["minute"]
    df["sin_minute"] = np.sin(2 * np.pi * minute_of_sol / 1440)
    df["cos_minute"] = np.cos(2 * np.pi * minute_of_sol / 1440)
    df["sin_hour"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]   = np.cos(2 * np.pi * df["hour"] / 24)
    return df

# ── Step 5: Normalize ─────────────────────────────────────────────────────────

def normalize(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Z-score normalize feature columns. Returns scaled df and scaler params dict.
    Fit on the full dataset here — we'll refit on train-only when building windows.
    """
    params = {}
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        mean = float(df[col].mean())
        std  = float(df[col].std())
        std  = std if std > 0 else 1.0   # avoid div by zero
        df[col] = (df[col] - mean) / std
        params[col] = {"mean": mean, "std": std}
    return df, params

# ── Step 6: Train/val/test split ──────────────────────────────────────────────

def split_by_sol(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    sols      = sorted(df["sol"].unique())
    n         = len(sols)
    train_end = sols[int(n * TRAIN_FRAC) - 1]
    val_end   = sols[int(n * (TRAIN_FRAC + VAL_FRAC)) - 1]

    train = df[df["sol"] <= train_end].reset_index(drop=True)
    val   = df[(df["sol"] > train_end) & (df["sol"] <= val_end)].reset_index(drop=True)
    test  = df[df["sol"] > val_end].reset_index(drop=True)

    splits = {
        "train_sol_range": [int(sols[0]),      int(train_end)],
        "val_sol_range":   [int(train_end + 1), int(val_end)],
        "test_sol_range":  [int(val_end + 1),   int(sols[-1])],
        "train_rows": len(train),
        "val_rows":   len(val),
        "test_rows":  len(test),
    }

    print("=== Train / Val / Test split ===")
    print(f"  Train: sols {splits['train_sol_range'][0]}–{splits['train_sol_range'][1]}  "
          f"({splits['train_rows']:,} rows)")
    print(f"  Val:   sols {splits['val_sol_range'][0]}–{splits['val_sol_range'][1]}  "
          f"({splits['val_rows']:,} rows)")
    print(f"  Test:  sols {splits['test_sol_range'][0]}–{splits['test_sol_range'][1]}  "
          f"({splits['test_rows']:,} rows)")
    print()

    return train, val, test, splits

# ── Step 7: Sliding window builder ───────────────────────────────────────────

def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays for supervised learning.

    X shape: (n_samples, window_size, n_features)
    y shape: (n_samples, n_targets)

    Each sample:
      X[i] = rows [i : i+window_size]          ← input window
      y[i] = target columns at row i+window_size+horizon-1  ← forecast target

    Important: windows do NOT cross sol boundaries to avoid
    treating the end of one sol as continuous with the start of the next.
    """
    feature_arr = df[feature_cols].values.astype(np.float32)
    target_arr  = df[target_cols].values.astype(np.float32)
    sol_arr     = df["sol"].values

    X_list, y_list = [], []
    n = len(df)

    for i in range(n - window_size - horizon + 1):
        window_end   = i + window_size
        target_idx   = window_end + horizon - 1

        # Skip if window or target crosses a sol boundary
        if sol_arr[i] != sol_arr[window_end - 1]:
            continue
        if sol_arr[window_end - 1] != sol_arr[target_idx]:
            continue
        # Skip windows containing any NaN
        if np.isnan(feature_arr[i:window_end]).any():
            continue
        X_list.append(feature_arr[i:window_end])
        y_list.append(target_arr[target_idx])

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_and_report(PARQUET_FILE)

    # 2. Gap report
    gap_report(df)

    # 3. Clean
    df = clean(df)

    # 4. Time features
    df = add_time_features(df)

    # Final feature list including cyclical time encodings
    all_features = FEATURE_COLS + ["sin_minute", "cos_minute", "sin_hour", "cos_hour"]
    all_features = [f for f in all_features if f in df.columns]

    # 5. Split BEFORE normalizing (prevent data leakage)
    train_df, val_df, test_df, splits = split_by_sol(df)
    del df
    gc.collect()

    # 6. Fit scaler on train only, apply to all splits
    print("=== Normalizing ===")
    train_df, scaler_params = normalize(train_df, all_features)

    # Apply train scaler to val and test
    for split_df in [val_df, test_df]:
        for col, p in scaler_params.items():
            if col in split_df.columns:
                split_df[col] = (split_df[col] - p["mean"]) / p["std"]

    print(f"  Scaler fit on {splits['train_rows']:,} train rows")
    print(f"  {len(scaler_params)} columns normalized\n")

    # Save scaler params for inference / inverse transform later
    with open(OUT_DIR / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"Saved scaler_params.json")

    # Save splits metadata
    with open(OUT_DIR / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits.json")

    # Save scaled dataframes
    scaled = pd.concat([train_df, val_df, test_df]).sort_values(["sol", "hour", "minute"])
    scaled.to_parquet(OUT_DIR / "insight_scaled.parquet", index=False)
    print(f"Saved insight_scaled.parquet\n")

    # 7. Build sliding windows
    print("=== Building sliding windows ===")
    print(f"  Window size:      {WINDOW_SIZE} min ({WINDOW_SIZE} timesteps)")
    print(f"  Forecast horizon: {FORECAST_HORIZON} min ahead")
    print(f"  Features:         {all_features}")
    print(f"  Targets:          {TARGET_COLS}\n")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  Building {name} windows...", end=" ", flush=True)
        X, y = build_windows(split_df, all_features, TARGET_COLS, WINDOW_SIZE, FORECAST_HORIZON)
        np.save(OUT_DIR / f"X_{name}.npy", X)
        np.save(OUT_DIR / f"y_{name}.npy", y)
        print(f"X={X.shape}  y={y.shape}")
        del X, y
        gc.collect()

    print(f"\nAll outputs saved to {OUT_DIR.resolve()}")
    print("\n=== Ready to train ===")
    print(f"  Load with:")
    print(f"    X_train = np.load('data/processed/X_train.npy')")
    print(f"    y_train = np.load('data/processed/y_train.npy')")
    print(f"    # X shape: (n_samples, {WINDOW_SIZE}, {len(all_features)})")
    print(f"    # y shape: (n_samples, {len(TARGET_COLS)})")


if __name__ == "__main__":
    main()
