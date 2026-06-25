# SleepSense — Production Data Collection & Ingestion Plan

This document details the production-level system design for collecting daytime behavioral metrics from user smartphones, processing them securely, sending them to the FastAPI backend, and presenting sleep predictions back to the user.

---

## 1. System Architecture Overview

```
 ┌────────────────────────────────────────────────────────┐
 │                   SmartPhone (Edge)                    │
 │ ┌───────────────────┐        ┌───────────────────────┐ │
 │ │  OS Sensor APIs   │        │ Local Database (Room/ │ │
 │ │ (CoreMotion, etc) │ ────►  │   SQLite) & Parser    │ │
 │ └───────────────────┘        └──────────┬────────────┘ │
 │                                         │ Aggregates   │
 │                                         ▼              │
 │                              ┌───────────────────────┐ │
 │                              │  Edge Processor (NLP  │ │
 │                              │  embeddings + JSON)   │ │
 │                              └──────────┬────────────┘ │
 └─────────────────────────────────────────┼──────────────┘
                                           │ POST /predict
                                           ▼
 ┌────────────────────────────────────────────────────────┐
 │                    Cloud Backend                       │
 │      ┌──────────────────────────────────────────┐      │
 │      │        FastAPI Prediction Server         │      │
 │      │  (XGBoost Model + PyTorch Autoencoder)   │      │
 │      └────────────────────┬─────────────────────┘      │
 │                           │                            │
 │                           ▼                            │
 │            SQLite / PostgreSQL Database                │
 │       (stores prediction history & anomalies)          │
 └───────────────────────────┬────────────────────────────┘
                             │ Returns predictions & advice
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │                      Client UI                         │
 │     Dashboard Widget / Push Notification / Widgets     │
 └────────────────────────────────────────────────────────┘
```

---

## 2. On-Device Sensor Data Collection (Mobile Level)

To collect the variables needed for the predict body, the mobile app uses native OS background listeners:

### A. Phone Lock/Unlock (Screen Time)
* **Android**: Register a background broadcast receiver listening for:
  - `Intent.ACTION_SCREEN_ON` (increment session count, start session timer).
  - `Intent.ACTION_SCREEN_OFF` (calculate session duration, check time-of-day).
* **iOS**: Track screen interactions using `UIApplicationDelegate` notifications or background notification pushes. Last lock hour acts as a bedtime proxy.

### B. Physical Activity Tracking
* **Android**: Use **Google Activity Recognition API** (`ActivityRecognitionClient`) which classifies states (stationary, walking, running) via accelerometer signals using on-device ML.
* **iOS**: Use the **CoreMotion Framework** (`CMMotionActivityManager`). Query historical activity logs using `queryActivityStarting(from:to:toQueue:withHandler:)` to aggregate daily walking, running, and stationary minutes.

### C. Foreground App Usage
* **Android**: Query `UsageStatsManager` (requires user to grant `PACKAGE_USAGE_STATS` permission) to capture the packages running in the foreground and sum up social, study, and entertainment app categories after 8 PM.
* **iOS**: Sandboxed apps cannot access system-wide app usage. 
  - *Production Workaround*: Use Apple's **Screen Time API** (specifically the `FamilyControls` and `DeviceActivity` frameworks) to monitor device activity categories in a privacy-preserving way.

### D. Audio & Conversation Proxy
* **Privacy-First Design**: Raw audio files **must never** leave the device.
* **Android & iOS**: Periodically trigger the microphone (e.g., for 10 seconds every 3 minutes) in the background. Use a local on-device Fast Fourier Transform (FFT) or a lightweight voice-activity detector (VAD) to classify sound into `silence` vs. `human voice`. Convert these checks to a daily ratio.

### E. GPS & Location Entropy
* **Android**: Use `FusedLocationProviderClient` with background geofences.
* **iOS**: Use `CLLocationManager` with `startMonitoringSignificantLocationChanges()`.
* **On-device clustering**: Run a lightweight DBSCAN clustering algorithm on the collected coordinate points. Calculate the distance variance (`mobility_radius`) and the unique clusters visited (`unique_locations_count`).

### F. Daily Diaries (NLP Notes)
* Present a quick "Daily Reflection" notepad inside the mobile UI at 9:00 PM.
* **On-device NLP**: Use a local Hugging Face `onnx` runtime or a lightweight static word embedding dictionary to preprocess notes text and extract similarity scores on-device, preserving user privacy.

---

## 3. Data Sync & API Transmission

1. **Daily Aggregation**: Every night at 10:00 PM, a background scheduler (e.g., **WorkManager** on Android, **BackgroundTasks** on iOS) compiles the accumulated sensors data into the single 42-feature JSON object.
2. **API Payload**:
   ```json
   {
     "unlock_count_late_night": 2.0,
     "walking_minutes": 45.0,
     "stress_level": 2.0,
     "nlp_caffeine_similarity": 0.12,
     ...
   }
   ```
3. **POST Request**: The background task sends an authenticated `POST` request to `https://api.sleepsense.io/predict` (our FastAPI backend).

---

## 4. Processing on FastAPI Backend

Upon receiving the payload, the backend:
1. Validates the JSON schema via **Pydantic**.
2. Aligns column sequence matching the trained **XGBoost** model.
3. Scales features using `feature_scaler.pkl` and predicts the sleep score.
4. Uses the **PyTorch Autoencoder** (`autoencoder_model.pkl`) to evaluate reconstruction loss. If the loss exceeds the user's historical 90th percentile threshold, flags the day as `anomaly_flag: 1`.
5. Feeds top SHAP attributions into **DistilGPT2 LLM** (with Chain-of-Thought prompting) to output exactly 3 personalized sleep advice recommendations.
6. Saves the prediction details, SHAP scores, and advice list to the **SQLite Database** (`predictions` table).

---

## 5. Rendering Output to the User

Once the backend responds, the client UI presents the results in three formats:

1. **Push Notification (Evening)**:
   - At 10:15 PM, send a push notification showing: 
     - *“Tonight’s predicted sleep: **Fairly Good**. Your late-night phone pickups are low today—keep it up!”*
2. **Dashboard Sleep Gauge (Morning Reflection)**:
   - When the user wakes up, the home dashboard displays:
     - An animated sleep gauge (color-coded from indigo to gold).
     - The top 3 recommendations generated by the LLM (e.g., *"Try box breathing before bed to reduce stress"*).
3. **Weekly Analytics Dashboard**:
   - Time-series charts comparing predicted vs. actual ratings.
   - An anomaly calendar heatmap showing days where their routine was severely broken (e.g., exam weeks).
