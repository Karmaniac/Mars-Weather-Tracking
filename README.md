# Storm Chasing on Mars

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Data Source](https://img.shields.io/badge/Data-NASA%20InSight-red)](https://atmos.nmsu.edu/data_and_services/atmospheres_data/INSIGHT/insight.html#Data)
[![Status](https://img.shields.io/badge/Project-Active-success)](#)


This repository contains processed and raw atmospheric data collected on the Martian surface by NASA instruments.  
The project focuses on pressure, temperature, and wind measurements, with the long-term goal of identifying significant atmospheric events and evaluating machine learning approaches for automated detection.

---

## Data Sources

All data originates from NASA’s **InSight lander** surface instruments:

- **PS (Pressure Sensor)**  
  Measures atmospheric pressure on Mars.

- **TWINS (Temperature and Wind for InSight)**  
  Measures:
  - Air temperature  
  - Wind speed and direction  

---

## Repository Structure
├── rawcsv/  
│ ├── ps/ # Raw pressure sensor CSV data  
│ └── twins/ # Raw temperature and wind CSV data  
│  
├── ConfigureCSV.ipynb  
├── combined_minute_avg.csv  
├── ps_calib_minute_avg.csv  
├── twins_calib_minute_avg.csv  
│  
├── pressure and temperature vs time.png  
└── README.md  


---

## Raw Data

The `rawcsv/` directory contains approximately **120 days of raw CSV data** used for:

- Data validation  
- Pipeline testing  
- Calibration and aggregation verification  

Subdirectories correspond to individual instruments:
- `ps/` → Pressure sensor data  
- `twins/` → Temperature and wind data  

---

## Data Processing Pipeline

### `ConfigureCSV.ipynb`

This Jupyter notebook serves as the primary data processing pipeline. It performs:

1. Cleaning of raw instrument data  
2. Temporal alignment of measurements  
3. Conversion into human-readable, minute-averaged CSV files  

---

## Generated Data Products

The following CSV files are generated when `ConfigureCSV.ipynb` is executed:

- **`ps_calib_minute_avg.csv`**  
  Minute-averaged and calibrated pressure data

- **`twins_calib_minute_avg.csv`**  
  Minute-averaged and calibrated temperature and wind data

- **`combined_minute_avg.csv`**  
  **Final cleaned dataset** combining pressure, temperature, and wind measurements into a single, analysis-ready file

---

## Visualizations

- **`pressure and temperature vs time.png`**  
  A validation plot illustrating the relationship between pressure and temperature

---

## Future Work

- [ ] Add labels for rapid pressure drop events  
- [ ] Add labels for significant temperature fluctuations  
- [ ] Research and compare machine learning models suitable for time-series anomaly detection  
- [ ] Develop and train a model to identify atmospheric events automatically  
