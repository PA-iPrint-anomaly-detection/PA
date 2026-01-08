# Predictive Maintenance for Printheads with Machine Learning

## Introduction

This repository contains the notebooks and final report for the **PA (Projet d’approfondissement)** titled **“Predictive maintenance using machine learning for printheads.”** The work was carried out at **HEIA-FR** in collaboration with **iPrint**, and investigates how to use **acoustic self-sensing** signals from piezoelectric inkjet printheads to detect and anticipate nozzle failures.

The pipeline isolates the informative **ring-down** portion of each waveform, fits a **physically interpretable damped-sinusoid** model, and uses the resulting **six fitted parameters** as compact features for:
- **Nozzle-state classification** (jetting vs. non-jetting / missing line)
- **Predictive maintenance** (forecasting eventual failure at different horizons)
- **Exploratory pressure regression** (predicting meniscus pressure from acoustic features)

For full methodology, experiments, and discussion, see **`PA_rapport_final.pdf`**.

## Repository contents

This repo is notebook-centric: each notebook corresponds to a major step in the workflow.

| Notebook | Purpose | Main outputs |
|---|---|---|
| **preprocessing.ipynb** | Build the 3D acoustic tensor from raw replicate data; basic waveform visualization across pressures. | Saved tensors + initial sanity plots. |
| **extract_ringdown.ipynb** | Crop/Extract the ring-down segment from each waveform. | Ring-down arrays + evolution plots across pressure. |
| **fitting_loop_merge_labels.ipynb** | Fit the damped-sinusoid model on ring-downs; apply phase/bank correction; merge partial fits; merge ISIS labels; includes an interactive viewer. | Fitted parameters per nozzle×pressure + merged NPZ (fits + labels) + interactive browsing. |
| **analysis_of_fits.ipynb** | Exploratory analysis of fitted parameters and fit errors vs pressure and labels; outlier summaries; jet→defect→jet episode analysis; interactive nozzle label map. | Interactive plots (scatter/mean±std/boxplots/maps) and label-conditioned summaries. |
| **binary_classification.ipynb** | Train and evaluate a **HistGradientBoostingClassifier** for **Jetting vs Non-jetting** (binary). | Group-aware split metrics, confusion matrices, optional distance-correlation analysis. |
| **steps_before_failure.ipynb** | Predict “eventual failure” K steps ahead (multi-horizon evaluation). | Confusion matrices and metrics per K; compact F1 vs K plot(s). |
| **predict_pressure.ipynb** | Exploratory regression: predict **pressure** from fitted ring-down parameters. | Ridge + HistGBRegressor training with grouped CV; test metrics; per-pressure mean prediction plot. |

## How to run

### 1) Dependencies
The notebooks were originally written for **Google Colab**, but can run locally with Python 3 if you install the usual scientific stack:

- `numpy`, `pandas`
- `scipy`
- `matplotlib` (and optionally `seaborn`)
- `scikit-learn`
- `plotly` (for interactive visualizations)
- optional: `joblib`, `tqdm`

### 2) Data paths
Most notebooks assume data living on Google Drive (`/content/drive/...`).  
If you run locally, **edit the path variables at the top of each notebook** so they point to your dataset (raw signals, intermediate tensors, fit outputs, and label files).

### 3) Recommended workflow (end-to-end)
1. **`preprocessing.ipynb`**: build acoustic tensor(s) from raw signals.
2. **`extract_ringdown.ipynb`**: isolate ring-down segments.
3. **`fitting_loop_merge_labels.ipynb`**: fit the parametric model + merge partials + merge ISIS labels.
4. **`analysis_of_fits.ipynb`**: explore parameter behavior, outliers, label structure, and spatial maps.
5. **`binary_classification.ipynb`**: jetting vs non-jetting classifier.
6. **`steps_before_failure.ipynb`**: failure forecasting at multiple horizons K.
7. **`predict_pressure.ipynb`**: exploratory pressure regression.

## Key takeaways (high level)

- **Ring-down modeling:** Fitting a damped-sinusoid yields compact, interpretable features per nozzle/pressure; phase/bank correction makes parameters comparable across banks.
- **Nozzle-state classification:** A histogram-based gradient boosting classifier achieves very strong separation between **jetting** and **missing line (non-jetting)**, with mistakes mostly being conservative false alarms.
- **Predictive maintenance:** Early-warning performance is **horizon-dependent**: the signal is strongest very close to the onset of failure and degrades quickly for longer horizons.
- **Structure & confounding:** Defects show spatial/pressure structure (e.g., clusters by columns, pressure-range extremes). Backpressure can explain a substantial part of acoustic variability, which is an important confound when interpreting generalization.
