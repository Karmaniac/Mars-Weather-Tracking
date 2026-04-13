# InSight Mars Weather Forecaster

An LSTM that predicts atmospheric pressure and temperature 60 minutes ahead using NASA InSight lander sensor data.

## Pipeline

Run the four scripts in order:

```bash
python download.py   # Fetch TWINS and PS CSVs from NMSU PDS archive
python process.py    # Merge and minute-average into a single parquet file
python prepare.py    # Normalize, split, and build sliding-window arrays
python train.py      # Train the LSTM and evaluate on the test set
```

## Data

- **TWINS** — wind speed, air temperature, rod temperature
- **PS** — atmospheric pressure and pressure temperature
- Sourced from the [NMSU PDS InSight archive](https://atmos.nmsu.edu/PDS/data/PDS4/InSight/)

## Model

A 2-layer stacked LSTM (256 hidden units) trained with a 120-minute input window to predict 60 minutes ahead. Targets: `PRESSURE` and `BMY_AVE_ROD_TEMP`.

## Outputs

| Path | Description |
|------|-------------|
| `data/raw/` | Downloaded CSVs |
| `data/processed/` | Merged parquet, scaled arrays, scaler params |
| `models/best_model.pt` | Best checkpoint by validation loss |
| `models/training_curve.png` | Loss plot |

## Requirements

```bash
pip install requests beautifulsoup4 pandas numpy pyarrow torch matplotlib
```