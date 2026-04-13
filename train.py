"""
4_train.py
----------
Trains an LSTM to forecast PRESSURE and BMY_AVE_ROD_TEMP 60 minutes ahead
using a 120-minute sliding window of InSight sensor data.

Outputs:
    models/best_model.pt        ← best checkpoint by val loss
    models/final_model.pt       ← model at end of training
    models/training_log.json    ← loss history per epoch
    models/training_curve.png   ← loss plot
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR  = Path("data/processed")
MODEL_DIR = Path("models")

# Architecture
INPUT_SIZE   = 8        # number of features (update if you changed FEATURE_COLS)
HIDDEN_SIZE  = 256      # LSTM hidden units
NUM_LAYERS   = 2        # stacked LSTM layers
DROPOUT      = 0.2      # dropout between LSTM layers
OUTPUT_SIZE  = 2        # PRESSURE, BMY_AVE_ROD_TEMP

# Training
BATCH_SIZE   = 512
EPOCHS       = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-5     # L2 regularisation
PATIENCE     = 10        # early stopping: stop if val loss doesn't improve

# Mixed precision — speeds up training on your RTX 5070 significantly
USE_AMP = True

# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("CUDA not available, using CPU")
    return dev

# ── Model ─────────────────────────────────────────────────────────────────────

class WeatherLSTM(nn.Module):
    """
    Stacked LSTM forecaster.

    Input:  (batch, seq_len, input_size)
    Output: (batch, output_size)  ← prediction at t + horizon
    """
    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        num_layers:  int,
        output_size: int,
        dropout:     float,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take only the last timestep's hidden state
        last = lstm_out[:, -1, :]
        return self.head(last)

# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(name: str, device: torch.device) -> TensorDataset:
    X = np.load(DATA_DIR / f"X_{name}.npy")
    y = np.load(DATA_DIR / f"y_{name}.npy")
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    print(f"  {name:5s}  X={tuple(X_t.shape)}  y={tuple(y_t.shape)}")
    return TensorDataset(X_t, y_t)

# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler:    torch.cuda.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        pred    = model(X_batch)
        loss    = criterion(pred, y_batch)
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)

# ── Inverse transform for reporting ──────────────────────────────────────────

def load_scaler(target_cols: list[str]) -> dict:
    with open(DATA_DIR / "scaler_params.json") as f:
        params = json.load(f)
    return {col: params[col] for col in target_cols if col in params}

def inverse_transform(value: float, col: str, scaler: dict) -> float:
    p = scaler[col]
    return value * p["std"] + p["mean"]

# ── Plot ──────────────────────────────────────────────────────────────────────

def save_loss_plot(log: dict, path: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(log["train_loss"], label="Train loss")
    plt.plot(log["val_loss"],   label="Val loss")
    if log.get("best_epoch") is not None:
        plt.axvline(log["best_epoch"], color="green", linestyle="--",
                    label=f"Best epoch ({log['best_epoch']+1})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (normalized)")
    plt.title("InSight Weather LSTM — Training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curve to {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

TARGET_COLS = ["PRESSURE", "BMY_AVE_ROD_TEMP"]

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n=== Loading windows ===")
    train_ds = load_split("train", device)
    val_ds   = load_split("val",   device)
    test_ds  = load_split("test",  device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WeatherLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(model)

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )
    criterion = nn.MSELoss()
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n=== Training ({EPOCHS} epochs, early stopping patience={PATIENCE}) ===\n")

    log = {"train_loss": [], "val_loss": [], "best_epoch": None}
    best_val_loss  = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
        val_loss   = eval_epoch(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        marker   = " ✓" if improved else ""

        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"({elapsed:.1f}s){marker}")

        if improved:
            best_val_loss = val_loss
            log["best_epoch"] = epoch
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   val_loss,
                "config": {
                    "input_size":  INPUT_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers":  NUM_LAYERS,
                    "output_size": OUTPUT_SIZE,
                    "dropout":     DROPOUT,
                }
            }, MODEL_DIR / "best_model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    # ── Save final model ──────────────────────────────────────────────────────
    torch.save({
        "epoch":       epoch,
        "model_state": model.state_dict(),
        "val_loss":    val_loss,
        "config": {
            "input_size":  INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers":  NUM_LAYERS,
            "output_size": OUTPUT_SIZE,
            "dropout":     DROPOUT,
        }
    }, MODEL_DIR / "final_model.pt")

    with open(MODEL_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    save_loss_plot(log, MODEL_DIR / "training_curve.png")

    # ── Evaluate best model on test set ──────────────────────────────────────
    print("\n=== Test set evaluation (best model) ===")
    checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Loaded best model from epoch {checkpoint['epoch']+1} "
          f"(val loss={checkpoint['val_loss']:.6f})")

    test_loss = eval_epoch(model, test_loader, criterion, device)
    print(f"  Test MSE (normalized): {test_loss:.6f}")

    # RMSE in real units using scaler
    scaler = load_scaler(TARGET_COLS)

    # Collect all predictions and targets
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)

    print("\n  Per-target RMSE (real units):")
    for i, col in enumerate(TARGET_COLS):
        if col not in scaler:
            continue
        std = scaler[col]["std"]
        rmse_normalized = np.sqrt(np.mean((preds[:, i] - targets[:, i]) ** 2))
        rmse_real = rmse_normalized * std
        unit = "Pa" if "PRESSURE" in col else "K"
        print(f"    {col:<30}  RMSE = {rmse_real:.4f} {unit}")

    print(f"\nAll model files saved to {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
