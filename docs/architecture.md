# SleepSense — Architecture

## Overview

SleepSense is a passive sleep quality predictor. It ingests daytime behavioral
signals from smartphone sensors and EMA surveys, and predicts tonight's sleep quality
score — without any wearable device.

## Layers

### Client Layer
- **Streamlit Dashboard** (Phase 1) — interactive web UI
- **React / Vite** (Phase 2 upgrade, on request)
- **Future: Native mobile app** — would call the same REST API with no backend changes

### API Gateway — FastAPI
- Multi-user endpoints with Pydantic validation
- Versioned routes (`/v1/...` ready)
- CORS configured for local and deployed origins

### AI Services Layer
| Service | Description |
|---------|-------------|
| **Prediction Service** | XGBoost + Random Forest regression → sleep quality score 0–3 |
| **Advice Service** | DistilGPT2 LLM + SHAP attribution → 3 empathetic recommendations |
| **Anomaly Detector** | Isolation Forest + PyTorch Deep Autoencoder per user → flags behaviorally unusual days |

### Data & Model Layer
- **SQLite database** (`models/registry/sleepsense.db`) — users, features, predictions
- **Model Registry** (`models/registry/`) — serialized model `.pkl` files
- **Parquet Cache** (`data/preprocessed/`) — computed feature vectors

### ETL Batch Pipeline
```
data/raw/ → src/data/loader.py → src/data/preprocessor.py
         → src/features/*.py  → src/data/feature_store.py
         → data/preprocessed/ → SQLite DB
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| SQLite (not MongoDB) | 49 users × ~70 days = ~3,500 rows; zero setup overhead |
| XGBoost regression | Best accuracy on tabular sensor data; SHAP-compatible |
| Leave-one-user-out CV | Most realistic evaluation for small N user studies |
| Streamlit first | Fastest path to a working demo; clean upgrade path to React |
| EMA daily labels | ~1,500+ training samples vs. 90 from PSQI; directly aligned with goal |
| No mobile app needed | REST contract is mobile-ready; web demo sufficient for academic scope |

## Data Flow

```
User Day D:
  Sensing data (activity, phonelock, audio, GPS, apps)
       ↓
  Feature Extractors (src/features/)
       ↓
  Daily Feature Vector (~30 features)
       ↓
  XGBoost Model → Score 0.0–3.0
       ↓
  Ordinal Post-rounding → Label (Very good / Fairly good / Fairly bad / Very bad)
       ↓
  SHAP Attribution → Top 3 driving features
       ↓
  Advice Generator (DistilGPT2 LLM Prompt) → 3 actionable recommendations
       ↓
  Stored in SQLite → Served via FastAPI → Displayed in Streamlit
```
