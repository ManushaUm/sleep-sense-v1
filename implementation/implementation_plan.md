# SleepSense — Passive Sleep Quality Predictor
# Implementation Plan (v3 — FINAL)

---

## Project Overview

SleepSense predicts **tonight's sleep quality** from today's daytime behavioral signals — no wearables required.

- **Dataset:** StudentLife (Dartmouth, 2013) — 49 participants, ~10 weeks
- **Data modalities:** Passive smartphone sensing + EMA self-reports + validated surveys
- **Core hypothesis:** Daytime behaviors (phone use, physical activity, stress, social interaction, screen timing) reliably predict that night's sleep quality

---

## All Decisions (Locked)

| Decision | Choice |
|----------|--------|
| **Target variable** | Daily EMA `rate` field — ~1,500–2,000 training samples |
| **Target type** | Regression (0–3 continuous) with ordinal post-rounding to label |
| **PSQI role** | Static per-user feature input (not target) |
| **Frontend** | Streamlit (Phase 1) → React/Vite upgrade on request |
| **Backend** | FastAPI, multi-user |
| **Database** | SQLite + SQLAlchemy |
| **Data structure** | `data/raw/` (original) + `data/preprocessed/` (pipeline output) |
| **Python env** | `sleepsense-ai` conda environment |
| **Mobile app** | Not required — backend is mobile-ready by REST contract |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Client Layer                           │
│      Streamlit Dashboard ──► React/Vite (future upgrade)     │
│      (Future: native mobile app — same REST contract)        │
└───────────────────────┬──────────────────────────────────────┘
                        │ REST / JSON over HTTP
┌───────────────────────▼──────────────────────────────────────┐
│              API Gateway — FastAPI (multi-user)               │
│      Versioned endpoints, Pydantic schemas, CORS             │
└─────┬──────────────────┬─────────────────────┬──────────────┘
      │                  │                     │
┌─────▼────┐  ┌──────────▼──────┐  ┌──────────▼─────────────┐
│ Predict  │  │  Advice Service │  │   Feature Store API    │
│ Service  │  │  (SHAP → rules) │  │  (daily vector lookup) │
│ XGB + RF │  │                 │  │                        │
└─────┬────┘  └────────┬────────┘  └──────────┬─────────────┘
      │                │                       │
┌─────▼────────────────▼───────────────────────▼─────────────┐
│                Data & Model Layer                            │
│  SQLite DB — users, daily_features, predictions             │
│  Model Registry — XGBoost, RF, IsoForest (.pkl)             │
│  Parquet Cache — preprocessed feature vectors per user      │
└──────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────▼──────────────────────────────────────┐
│                 ETL Batch Pipeline                            │
│   data/raw/ ──► data/preprocessed/ ──► feature vectors ──► DB│
└──────────────────────────────────────────────────────────────┘
```

---

## Project Directory Structure

```
SleepSense/
├── data/
│   ├── raw/                        ✅ StudentLife original data
│   │   ├── EMA/
│   │   │   ├── EMA_definition.json
│   │   │   └── response/
│   │   │       ├── Sleep/          ← PRIMARY TARGET (EMA daily rate)
│   │   │       ├── Stress/
│   │   │       ├── Activity/
│   │   │       ├── Exercise/
│   │   │       └── ...
│   │   ├── app_usage/
│   │   ├── sensing/
│   │   │   ├── activity/
│   │   │   ├── phonelock/
│   │   │   ├── audio/
│   │   │   ├── conversation/
│   │   │   ├── dark/
│   │   │   ├── bluetooth/
│   │   │   ├── gps/
│   │   │   └── phonecharge/
│   │   └── survey/
│   │       ├── psqi.csv            ← STATIC USER FEATURE
│   │       ├── BigFive.csv
│   │       └── ...
│   ├── preprocessed/               ✅ Pipeline output
│   │   ├── features/               Daily feature CSVs per user
│   │   ├── targets/                EMA sleep labels aligned to dates
│   │   └── merged/                 Final model-ready dataset (.parquet)
│   └── Readme.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   EDA — distributions, gaps, timestamps
│   ├── 02_feature_engineering.ipynb Feature pipeline walkthrough + validation
│   ├── 03_model_training.ipynb     Baseline → RF → XGBoost, LOUO-CV results
│   └── 04_results_analysis.ipynb   SHAP plots, anomaly detection, error analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py               Raw data readers (one fn per modality)
│   │   ├── preprocessor.py         Cleaning, timestamp alignment, label extraction
│   │   └── feature_store.py        build_daily_features(user, date) → dict
│   ├── features/
│   │   ├── app_usage_features.py   App category extraction + timing features
│   │   ├── activity_features.py    Stationary ratio, walk/run time, bouts
│   │   ├── phonelock_features.py   Unlock counts, session lengths, late-night use
│   │   ├── audio_features.py       Silence ratio, conversation time
│   │   ├── gps_features.py         Location entropy, mobility radius
│   │   ├── notes_nlp.py            Word2Vec embeddings for daily notes text
│   │   └── ema_features.py         Stress, mood, social, exercise (self-report)
│   ├── models/
│   │   ├── baseline.py             Rule-based heuristic (sanity check)
│   │   ├── regression.py           RF + XGBoost training logic
│   │   ├── anomaly.py              Isolation Forest — per-user behavioral outlier
│   │   ├── anomaly_ae.py           PyTorch Deep Autoencoder for anomaly detection
│   │   └── trainer.py              Cross-validation, grid search, model saving
│   ├── evaluation/
│   │   ├── metrics.py              MAE, RMSE, binary acc, ordinal acc
│   │   └── explainability.py       SHAP TreeExplainer, waterfall + summary plots
│   ├── advice/
│   │   ├── generator.py            Wrapper for advice recommendations
│   │   └── llm_generator.py        Transformer LLM (DistilGPT2) with CoT and few-shot prompts
│   └── db/
│       ├── database.py             SQLite engine + session factory
│       ├── models.py               SQLAlchemy ORM: User, DailyFeatures, Prediction
│       └── crud.py                 get_user, store_features, store_prediction, etc.
│
├── app/
│   ├── api/
│   │   ├── main.py                 FastAPI app, CORS, lifespan hooks
│   │   ├── schemas.py              Pydantic: DailyFeaturesInput, PredictionResponse
│   │   └── routers/
│   │       ├── health.py           GET /health
│   │       ├── predict.py          POST /predict, POST /predict/{user_id}
│   │       ├── advice.py           GET /advice/{user_id}
│   │       └── history.py          GET /history/{user_id}, GET /anomalies/{user_id}
│   └── frontend/
│       ├── streamlit_app.py        Phase 1: Streamlit dashboard
│       └── react/                  Phase 2 (future): React + Vite app
│
├── models/
│   └── registry/
│       ├── sleepsense.db           SQLite database file
│       ├── xgboost_model.pkl
│       ├── rf_model.pkl
│       ├── isoforest_model.pkl
│       ├── autoencoder_model.pkl   PyTorch Deep Autoencoder checkpoint
│       ├── feature_scaler.pkl
│       ├── feature_importance.json
│       └── experiments.csv
│
├── implementation/
│   ├── implementation_plan.md      This file
│   └── progress.md                 Live progress tracker
│
├── docs/
│   └── architecture.md
│
├── requirements.txt
├── requirements-dev.txt
└── Readme.md
```

---

## Feature Catalog (~30 features per user per day)

### Phone Lock/Unlock — `sensing/phonelock/`
| Feature | Description | Sleep Relevance |
|---------|-------------|----------------|
| `unlock_count_late_night` | # unlocks after 22:00 | Strongest negative predictor |
| `last_unlock_hour` | Hour of final phone use | Bedtime proxy |
| `first_unlock_hour` | Hour of first unlock | Wake-up proxy |
| `unlock_count_evening` | # unlocks 20–22:00 | Wind-down quality |
| `unlock_count_daytime` | # unlocks 06–20:00 | Baseline engagement |
| `avg_session_duration_min` | Avg phone session length | Engagement intensity |

### Physical Activity — `sensing/activity/`
| Feature | Description | Sleep Relevance |
|---------|-------------|----------------|
| `stationary_ratio` | Fraction of time stationary | Sedentary → worse sleep |
| `walking_minutes` | Total walking time | Light activity |
| `running_minutes` | Total vigorous activity | Exercise → better sleep |
| `exercise_detected` | Boolean: any vigorous bout | Key binary predictor |
| `peak_activity_hour` | Hour with most movement | Timing of exercise |
| `activity_bout_count` | # distinct active periods | Fragmentation |

### App Usage — `app_usage/`
| Feature | Description | Sleep Relevance |
|---------|-------------|----------------|
| `app_social_min` | Social media (daytime) | Cognitive stimulation |
| `app_entertainment_evening_min` | Entertainment after 20:00 | Blue light + arousal |
| `app_late_night_min` | Any app use after 22:00 | Direct disruptor |
| `last_active_app_hour` | Last non-system app hour | 2nd bedtime proxy |
| `app_diversity_count` | Unique apps used | Cognitive load |
| `app_study_sessions` | Study/academic app use | Academic stress proxy |

### Audio / Social — `sensing/audio/`, `sensing/conversation/`
| Feature | Description | Sleep Relevance |
|---------|-------------|----------------|
| `silence_ratio` | Fraction in silence | Calm environment |
| `conversation_ratio` | Detected conversation | Social activity level |
| `social_audio_evening` | Conversation after 19:00 | Cortisol elevation |

### EMA Self-Report (daily surveys)
| Feature | Sleep Relevance |
|---------|----------------|
| `stress_level` (1–5) | #1 self-report predictor |
| `mood_happy`, `mood_tired` | Affect and fatigue indicators |
| `social_contacts` | Social load |
| `exercise_self_report` | Cross-validates sensing |

### Context Features
| Feature | |
|---------|--|
| `day_of_week`, `is_weekend` | Sleep rhythm patterns |
| `psqi_pre_score` | User baseline (static) |
| `study_hours_today` | Academic pressure |

---

## Model Design

### Primary: XGBoost Regressor
- Input: ~30 features (daily vector)
- Output: Score 0.0–3.0 → rounded → label:
  - 2.5–3.0 = **Very good**
  - 1.5–2.4 = **Fairly good**
  - 0.5–1.4 = **Fairly bad**
  - 0.0–0.4 = **Very bad**
- Validation: Leave-one-user-out (LOUO) cross-validation

### Supporting Models
| Model | Purpose |
|-------|---------|
| Random Forest | Feature importance, SHAP comparison |
| Ridge Regression | Linear baseline |
| Isolation Forest | Per-user anomaly detection |
| PyTorch Autoencoder | Reconstruction-based behavioral anomaly detection |

### Evaluation Targets
| Metric | Goal |
|--------|------|
| MAE | < 0.8 |
| RMSE | < 1.0 |
| Binary accuracy (Good vs Bad) | > 65% |
| Ordinal accuracy (within 1 label) | > 80% |

---

## Database Schema (SQLite)

```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    psqi_pre_score REAL,
    psqi_post_score REAL,
    personality_json TEXT
);

CREATE TABLE daily_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT REFERENCES users(user_id),
    date TEXT,                        -- 'YYYY-MM-DD'
    features_json TEXT,               -- all ~30 features as JSON blob
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, date)
);

CREATE TABLE predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT REFERENCES users(user_id),
    date TEXT,
    predicted_score REAL,
    predicted_label TEXT,
    anomaly_flag INTEGER DEFAULT 0,
    top_features_json TEXT,           -- top 3 SHAP drivers
    advice_json TEXT,                 -- list of suggestion strings
    actual_rating TEXT,               -- EMA ground truth (post-hoc)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, date)
);
```

---

## FastAPI Endpoints

```
GET  /health                      Liveness probe
GET  /users                       List all ingested users
POST /users/{user_id}/ingest      Run ETL + feature build for a user
POST /predict                     Predict from raw feature JSON (anonymous)
POST /predict/{user_id}           Predict using stored user features
GET  /advice/{user_id}            Today's advice for a user
GET  /history/{user_id}           Full prediction history + actuals
GET  /anomalies/{user_id}         Anomaly-flagged days for a user
```

---

## Streamlit Dashboard Pages

| Page | Contents |
|------|----------|
| Home | Animated sleep score gauge, today's prediction, top 3 advice |
| Daily Breakdown | Activity / phone / app / EMA bar charts by time-of-day |
| Trend View | 10-day predicted vs. actual sleep quality |
| Advice Panel | Personalized recommendations with SHAP explanations |
| User Explorer | Cross-user sleep quality distributions |
| Anomaly Log | Calendar heatmap of behaviorally anomalous days |

---

## Advice Engine (Transformer-based LLM + Prompt Engineering)

| Feature Driver | Advice |
|---------------|--------|
| `unlock_count_late_night` high | "You picked up your phone {N}× after 10 PM — try charging it outside your bedroom." |
| `stationary_ratio` > 0.9 | "You barely moved today. A 20-min walk improves sleep onset significantly." |
| `stress_level` > 3 | "High stress today. Try box breathing (4-4-4-4) before bed." |
| `app_entertainment_evening_min` high | "{N} mins of screens after 8 PM suppresses melatonin. Stop 1 hour before bed." |
| `running_minutes` = 0 | "No vigorous activity detected. Even a short jog tomorrow helps sleep." |
| `last_unlock_hour` > 23 | "Last phone use at {H}:00. A consistent phone-off time trains your rhythm." |
| `social_audio_evening` high | "Late social activity raises cortisol. Wind down earlier." |
| `study_hours_today` high + `stress_level` high | "Heavy study day — schedule tomorrow's work earlier to avoid bedtime anxiety." |
