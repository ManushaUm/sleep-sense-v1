# SleepSense 🌙

> **Passive Sleep Quality Predictor from Daytime Behavior**
> Predict tonight's sleep quality from your smartphone's daytime signals — no wearables required.

---

## What is SleepSense?

Most sleep research focuses on what happens *at night*. SleepSense argues that **your day predicts your night**.

The science is well-established: cortisol levels, physical activity, light exposure timing, caffeine-related behavior, and social interaction patterns during the day are all strong predictors of sleep onset latency and sleep quality that night. You don't need to watch someone sleep to know they'll sleep badly.

SleepSense uses the [StudentLife dataset](http://studentlife.cs.dartmouth.edu/) (Dartmouth College, 2013 — 49 participants, 10 weeks) to train a machine learning model that:
1. Reads daytime behavioral signals (phone usage timing, physical activity, stress, social interaction)
2. Predicts a **sleep quality score** (0–3 continuous, mapped to Very good / Fairly good / Fairly bad / Very bad)
3. Generates **actionable suggestions** to improve tonight's sleep — powered by SHAP explainability

---

## Architecture

```
Streamlit Dashboard / React (future)
         ↓ REST API
     FastAPI Backend (multi-user)
    ┌────────────────────────────┐
    │  Prediction (XGB + RF)     │
    │  Advice (DistilGPT2 LLM)   │
    │  Anomaly (IsoF. + PyTorch) │
    └──────────┬─────────────────┘
         SQLite + Model Registry
         ETL Pipeline
    data/raw/ → data/preprocessed/
```

See [`docs/architecture.md`](docs/architecture.md) for full detail.

---

## Dataset

**StudentLife** — Dartmouth College, 2013

| Modality | Files | Signal |
|----------|-------|--------|
| `data/raw/EMA/response/Sleep/` | 49 JSONs | **Target**: daily sleep quality rating |
| `data/raw/sensing/activity/` | 49 CSVs | Physical activity (stationary/walking/running) |
| `data/raw/sensing/phonelock/` | 49 CSVs | Phone lock/unlock → screen time + late-night use |
| `data/raw/sensing/audio/` | 49 CSVs | Ambient sound → conversation proxy |
| `data/raw/app_usage/` | 49 CSVs | Foreground app → social media, study, entertainment timing |
| `data/raw/sensing/gps/` | 49 CSVs | Location entropy → mobility |
| `data/raw/survey/psqi.csv` | 1 CSV | PSQI baseline (used as static user feature) |

---

## Setup

### Prerequisites
- [Anaconda](https://www.anaconda.com/) or Miniconda
- Python 3.10+

### 1. Activate the environment
```bash
conda activate sleepsense-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Dev) Install dev tools
```bash
pip install -r requirements-dev.txt
```

---

## Running the Project

### Run the EDA notebook
```bash
conda activate sleepsense-ai
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Run the Streamlit dashboard
```bash
conda activate sleepsense-ai
streamlit run app/frontend/streamlit_app.py
```

### Start the FastAPI backend
```bash
conda activate sleepsense-ai
uvicorn app.api.main:app --reload --port 8000
```

API docs available at: `http://localhost:8000/docs`

---

## Project Structure

```
SleepSense/
├── data/raw/              StudentLife original data (untouched)
├── data/preprocessed/     Pipeline output (feature vectors, targets, merged dataset)
├── notebooks/             Jupyter notebooks (EDA → features → models → results)
├── src/
│   ├── data/              Data loaders, preprocessor, feature store
│   ├── features/          Per-modality feature extractors
│   ├── models/            Baseline, XGBoost, Isolation Forest, trainer
│   ├── evaluation/        Metrics and SHAP explainability
│   ├── advice/            Advice generation engine
│   └── db/                SQLite ORM and CRUD helpers
├── app/
│   ├── api/               FastAPI backend (routers, schemas)
│   └── frontend/          Streamlit dashboard
├── models/registry/       Saved model artifacts + SQLite DB
├── implementation/        Project plan and progress tracker
└── docs/                  Architecture documentation
```

---

## Implementation Progress

See [`implementation/progress.md`](implementation/progress.md) for the live progress tracker.

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Target variable | Daily EMA sleep rating | ~1,500 samples vs. 90 from PSQI |
| Model | XGBoost regression (0–3) | Best accuracy on tabular data; SHAP-compatible |
| Explainability | SHAP TreeExplainer | Per-prediction top-3 feature attribution |
| Database | SQLite + SQLAlchemy | Zero-setup; trivial PostgreSQL migration |
| Frontend | Streamlit → React | Fast iteration then polish |
| Mobile app | Not required | REST API is mobile-ready by design |
