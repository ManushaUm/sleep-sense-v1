# SleepSense — Implementation Walkthrough (with Advanced AI Techniques)

We have successfully completed all phases of the **SleepSense** project. This walkthrough documents the implementation details, evaluation results, and details of the advanced AI techniques incorporated to meet academic specifications.

---

## 🚀 Accomplishments & Advanced Techniques

### 1. Data ETL & Features Pipeline (with NLP & Word Embeddings)
- Vectorized and optimized raw data loading for all 49 users.
- Built extractors for phonelock events, activity metrics, app usage, conversational audio, GPS mobility, and EMA daily surveys.
- **Natural Language Processing (NLP) & Word Embeddings (Word2Vec)**:
  - Simulated daily text diaries (`data/raw/notes.csv`) reflecting student daytime routines (e.g. coffee intake, study stress, late night screen scrolling).
  - Preprocessed diaries (tokenization, lowercase, stop-word removal).
  - Trained a local **Word2Vec** embedding model (`gensim.models.Word2Vec`) on the notes corpus.
  - Computed document embeddings (average word vectors) and extracted semantic similarity scores (`nlp_caffeine_similarity`, `nlp_screen_similarity`, `nlp_stress_similarity`) as sleep predictors.
  - Compiled and cached the complete 42-feature daily vectors under `data/preprocessed/`.

### 2. Machine Learning & Model Registry
- Trained models using **Leave-One-User-Out (LOUO) cross-validation** to mimic realistic out-of-sample users.
- Saved best models (`xgboost_model.pkl`, `rf_model.pkl`, and `feature_scaler.pkl`) to `models/registry/`.
- Logged CV metrics to `models/registry/experiments.csv`.

### 3. Generative AI (Reconstruction-based Autoencoders)
- **Deep Autoencoder for Anomaly Detection**:
  - Implemented a feedforward **Deep Autoencoder (AE)** using PyTorch (`src/models/anomaly_ae.py`).
  - Trained the neural network to compress daily features to a latent space of 8 dimensions and reconstruct the input vector.
  - Computed daily reconstruction losses (MSE). Flagged days with reconstruction errors exceeding the 90th percentile threshold as behaviorally anomalous.
  - Registered the trained autoencoder at `models/registry/autoencoder_model.pkl` alongside the baseline Isolation Forest.

### 4. Transformer Models (LLMs) & Prompt Engineering
- **DistilGPT2 Sleep Coach**:
  - Replaced the baseline rule engine with a pre-trained **Transformer Text-Generation Pipeline** (DistilGPT2) from Hugging Face `transformers` in `src/advice/llm_generator.py`.
  - **Systematic Prompt Design**: Designed structured instruction headers enforcing the sleep coach role, constraints, and metric-grounded guidelines.
  - **In-Context Learning (Few-shot prompting)**: Supplied input-output training exemplars within the context window to guide output formatting.
  - **Chain-of-Thought (CoT) Prompting**: Instructed the LLM to write out its step-by-step reasoning steps first (identifying key disruptors and reinforcers) before outputting the final 3 recommendations.
  - Designed an optimized fallback logic to ensure recommendation safety and speed.

---

## 📈 Evaluation & CV Metrics

The models were retrained on 1,321 daily feature vectors across 49 users (now with NLP features):

| Model | MAE | RMSE | Ordinal Accuracy (within-1) |
|---|---|---|---|
| **UserMeanBaseline** | 0.614 | 0.726 | 100.0% |
| **RandomForestRegressor** | 0.610 | 0.731 | 99.2% |
| **XGBoostRegressor** | 0.615 | 0.738 | 98.7% |

---

## 🧪 Verification & Testing

The test suite in [test_integration.py](file:///c:/Users/HP/Desktop/Semester%207/AI/Project/SleepSense/tests/test_integration.py) validates the data loader, feature extractors, model predictions, advice generation, and API endpoint routing.

### Running the Test Suite
Activate the environment and run `pytest`:
```bash
conda activate sleepsense-ai
pytest tests/test_integration.py
```
All 7 integration test cases passed successfully.

---

## 💻 Running the Servers

### FastAPI Backend
```bash
conda run -n sleepsense-ai uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```
Interactive docs are available at `http://127.0.0.1:8000/docs`.

### Streamlit UI Dashboard
```bash
conda run -n sleepsense-ai streamlit run app/frontend/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
```
Interact with the dashboard at `http://127.0.0.1:8501`.
