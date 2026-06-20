# SleepSense — Progress Tracker

> Last Updated: 2026-06-20

---

## Overall Status

```
Phase 1  — Environment & Project Skeleton    ✅ COMPLETE
Phase 2  — Data Exploration (EDA)            ✅ COMPLETE
Phase 3  — Data Loaders                      ✅ COMPLETE
Phase 4  — Feature Extractors                ✅ COMPLETE
Phase 5  — Feature Store + Preprocessed      ✅ COMPLETE
Phase 6  — Feature Engineering Notebook      ✅ COMPLETE
Phase 7  — Model Training                    ✅ COMPLETE
Phase 8  — SHAP Explainability               ✅ COMPLETE
Phase 9  — Anomaly Detection                 ✅ COMPLETE
Phase 10 — Advice Generator                  ✅ COMPLETE
Phase 11 — Database + FastAPI Backend        ✅ COMPLETE
Phase 12 — Streamlit Dashboard               ✅ COMPLETE
Phase 13 — Integration Testing + README      ✅ COMPLETE
```

---

## Phase 1 — Environment & Project Skeleton

**Goal:** Install all dependencies, scaffold all directories and `__init__.py` files, create `requirements.txt`.

- [ ] Create `requirements.txt`
- [ ] Install packages into `sleepsense-ai` conda env
- [ ] Scaffold `src/` directory structure
- [ ] Scaffold `app/` directory structure
- [ ] Create placeholder `__init__.py` files
- [ ] Create `data/preprocessed/features/`, `targets/`, `merged/` directories
- [ ] Create `models/registry/` directory

**Status:** ✅ Complete

**Installed (key packages):**
- pandas 3.0.3, numpy 2.4.6, pyarrow 24.0.0
- scikit-learn 1.9.0, xgboost 3.3.0, lightgbm 4.6.0, shap 0.52.0
- fastapi 0.138.0, uvicorn 0.49.0, pydantic 2.13.4, sqlalchemy 2.0.51
- streamlit 1.58.0, matplotlib 3.11.0, seaborn 0.13.2, plotly 6.8.0

---

## Phase 2 — Data Exploration (EDA)

**Goal:** Understand all data modalities, validate target distribution, identify missing data and coverage gaps.

- [x] Load and inspect `EMA/response/Sleep/` — check coverage per user
- [x] Visualize EMA sleep quality distribution (target variable balance)
- [x] Load and inspect `sensing/activity/` — check timestamps and label types
- [x] Load and inspect `sensing/phonelock/` — check event types
- [x] Load and inspect `app_usage/` — identify app package taxonomy
- [x] Load `survey/psqi.csv` — visualize user baseline scores
- [x] Check date alignment between EMA labels and sensing data
- [x] Identify users with insufficient data (< 10 EMA responses)
- [x] Document all findings in `notebooks/01_data_exploration.ipynb`

**Status:** ✅ Complete

---

## Phase 3 — Data Loaders

**Goal:** Build clean, tested data loading functions for every modality.

**File:** `src/data/loader.py`

- [x] `load_ema_sleep(user_id)` → DataFrame (date, hours, quality_label)
- [x] `load_activity(user_id)` → DataFrame (timestamp, activity_type)
- [x] `load_phonelock(user_id)` → DataFrame (timestamp, event_type)
- [x] `load_app_usage(user_id)` → DataFrame (timestamp, package_name)
- [x] `load_audio(user_id)` → DataFrame (timestamp, audio_type)
- [x] `load_conversation(user_id)` → DataFrame (timestamp, inferred_conversation)
- [x] `load_gps(user_id)` → DataFrame (timestamp, lat, lng)
- [x] `load_psqi()` → DataFrame (user_id, psqi_pre_score, psqi_post_score)
- [x] `load_ema_stress(user_id)` → DataFrame (date, stress_level)
- [x] `load_ema_exercise(user_id)` → DataFrame (date, exercise_bool, duration)
- [x] `get_all_users()` → list of user IDs with data

**File:** `src/data/preprocessor.py`

- [x] Timestamp normalization to UTC / local time
- [x] EMA label encoding (Very good=3, Fairly good=2, Fairly bad=1, Very bad=0)
- [x] Handle missing data and outlier timestamps

**Status:** ✅ Complete

---

## Phase 4 — Feature Extractors

**Goal:** One extractor module per modality — each takes (user_id, date) and returns a dict of features.

- [x] `src/features/phonelock_features.py`
  - [x] `unlock_count_late_night`, `unlock_count_evening`, `unlock_count_daytime`
  - [x] `first_unlock_hour`, `last_unlock_hour`
  - [x] `avg_session_duration_min`, `screen_sessions_count`

- [x] `src/features/activity_features.py`
  - [x] `stationary_ratio`, `walking_minutes`, `running_minutes`
  - [x] `exercise_detected`, `peak_activity_hour`, `activity_bout_count`

- [x] `src/features/app_usage_features.py`
  - [x] App category taxonomy (social, entertainment, study, browser, system)
  - [x] `app_social_min`, `app_entertainment_evening_min`, `app_late_night_min`
  - [x] `last_active_app_hour`, `app_diversity_count`, `app_study_sessions`

- [x] `src/features/audio_features.py`
  - [x] `silence_ratio`, `conversation_ratio`, `social_audio_evening`

- [x] `src/features/gps_features.py`
  - [x] `location_entropy`, `mobility_radius`, `unique_locations_count`

- [x] `src/features/ema_features.py`
  - [x] `stress_level`, `mood_happy`, `mood_tired`
  - [x] `social_contacts`, `exercise_self_report`, `study_hours_today`

**Status:** ✅ Complete

---

## Phase 5 — Feature Store + Preprocessed Output

**Goal:** Combine all extractors into a single pipeline that outputs `data/preprocessed/`.

**File:** `src/data/feature_store.py`

- [x] `build_daily_features(user_id, date)` → dict of all ~30 features
- [x] `build_dataset(user_ids, date_range)` → DataFrame (features + label)
- [x] Export per-user CSVs to `data/preprocessed/features/`
- [x] Export aligned labels to `data/preprocessed/targets/`
- [x] Export merged dataset to `data/preprocessed/merged/dataset.parquet`
- [x] Handle days where EMA label is missing (exclude from supervised set)

**Status:** ✅ Complete

---

## Phase 6 — Feature Engineering Notebook

**Goal:** Validate the feature pipeline, check correlations, visualize feature distributions.

**File:** `notebooks/02_feature_engineering.ipynb`

- [x] Feature distribution plots (per category)
- [x] Correlation heatmap with target
- [x] Missing value analysis per feature
- [x] User-level variance analysis
- [x] Class balance check (target label distribution)

**Status:** ✅ Complete

---

## Phase 7 — Model Training

**Goal:** Train and evaluate baseline, RF, and XGBoost models. Save best model.

- [x] `src/models/baseline.py` — rule-based heuristic
- [x] `src/models/regression.py` — RF + XGBoost
- [x] `src/models/trainer.py` — LOUO cross-validation, grid search
- [x] `notebooks/03_model_training.ipynb` — model comparison table
- [x] Save best model → `models/registry/xgboost_model.pkl`
- [x] Save feature scaler → `models/registry/feature_scaler.pkl`
- [x] Log results → `models/registry/experiments.csv`

**Model targets:**
- MAE < 0.8 | RMSE < 1.0 | Binary acc > 65% | Ordinal acc (within-1) > 80%

**Status:** ✅ Complete

---

## Phase 8 — SHAP Explainability

**Goal:** Generate SHAP values for all predictions; save global feature importance.

**File:** `src/evaluation/explainability.py`

- [x] SHAP TreeExplainer setup for XGBoost
- [x] Per-prediction: top-3 SHAP contributors
- [x] Global: summary plot + bar chart
- [x] Save `models/registry/feature_importance.json`
- [x] Add SHAP analysis to `notebooks/04_results_analysis.ipynb`

**Status:** ✅ Complete

---

## Phase 9 — Anomaly Detection

**Goal:** Flag days where a user's behavior deviates significantly from their own baseline.

**File:** `src/models/anomaly.py`

- [x] Fit Isolation Forest per user on their feature history
- [x] Compute anomaly score for each (user, day)
- [x] Threshold: flag top 10% most anomalous days
- [x] Save `models/registry/isoforest_model.pkl`
- [x] Add anomaly flags to `predictions` table

**Status:** ✅ Complete

---

## Phase 10 — Advice Generator

**Goal:** Map top SHAP features to actionable sleep advice strings.

**File:** `src/advice/generator.py`

- [x] Rule library (feature → advice string with dynamic values)
- [x] `generate_advice(shap_top3, feature_values)` → list of 3 advice strings
- [x] Cover all major feature categories
- [x] Test advice generation for edge cases

**Status:** ✅ Complete

---

## Phase 11 — Database + FastAPI Backend

**Goal:** Persist predictions and expose REST API.

- [x] `src/db/database.py` — SQLite engine + session
- [x] `src/db/models.py` — SQLAlchemy ORM models
- [x] `src/db/crud.py` — CRUD helpers
- [x] `app/api/main.py` — FastAPI app setup
- [x] `app/api/schemas.py` — Pydantic input/output models
- [x] `app/api/routers/health.py`
- [x] `app/api/routers/predict.py`
- [x] `app/api/routers/advice.py`
- [x] `app/api/routers/history.py`
- [x] API smoke tests (all endpoints return 200)

**Status:** ✅ Complete

---

## Phase 12 — Streamlit Dashboard

**Goal:** Interactive web dashboard with 6 pages.

**File:** `app/frontend/streamlit_app.py`

- [x] Home page — sleep score gauge + prediction card + advice
- [x] Daily Breakdown — feature charts by time-of-day
- [x] Trend View — 10-day predicted vs. actual chart
- [x] Advice Panel — top 3 recommendations with SHAP explanation
- [x] User Explorer — cross-user comparison
- [x] Anomaly Log — calendar heatmap

**Status:** ✅ Complete

---

## Phase 13 — Integration Testing + README

- [x] End-to-end pipeline test (raw data → prediction → advice → UI)
- [x] Unit tests for critical feature extractors
- [x] API integration tests
- [x] Update `Readme.md` with setup instructions, architecture, and usage
- [x] Final review of `data/Readme.md`

**Status:** ✅ Complete

---

## Decisions Log

| Date | Decision |
|------|----------|
| 2026-06-19 | Dataset confirmed: StudentLife (Dartmouth 2013) |
| 2026-06-19 | Target: Daily EMA `rate` field (Option A) |
| 2026-06-19 | Model output: Regression 0–3 with ordinal post-rounding |
| 2026-06-19 | Frontend: Streamlit first, React on request |
| 2026-06-19 | Database: SQLite + SQLAlchemy (no MongoDB) |
| 2026-06-20 | Mobile app: Not required — backend is mobile-ready by REST contract |
| 2026-06-20 | Data structure: data/raw/ + data/preprocessed/ confirmed |

---

## Notes & Issues

_Add any blockers, data quality issues, or design decisions discovered during implementation here._

