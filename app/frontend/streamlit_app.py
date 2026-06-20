import os
import sys
import requests
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Configure Page
st.set_page_config(
    page_title="SleepSense — Passive Sleep Quality Predictor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = "http://127.0.0.1:8000"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "preprocessed"
REGISTRY_DIR = Path(__file__).resolve().parents[2] / "models" / "registry"

# Custom Styling (wow aesthetics)
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
        color: #f0f2f6;
    }
    .stApp {
        background-color: #0b0f19;
    }
    .sidebar .sidebar-content {
        background-color: #0f1626;
    }
    h1, h2, h3 {
        color: #6366f1 !important;
        font-family: 'Outfit', sans-serif;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .advice-card {
        background: rgba(30, 41, 59, 0.3);
        border-left: 5px solid #6366f1;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .anomaly-banner {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ── API Client / Fallback Data Loader ──────────────────────────────────────────

@st.cache_data
def get_user_list():
    """Get list of user IDs."""
    try:
        r = requests.get(f"{API_URL}/users", timeout=2.0)
        if r.status_code == 200:
            return [u['user_id'] for u in r.json()]
    except Exception:
        pass
    
    # Fallback to local files
    feat_dir = DATA_DIR / "features"
    if feat_dir.exists():
        users = [f.name.replace("_features.csv", "") for f in feat_dir.glob("u*_features.csv")]
        return sorted(users)
    return ["u00", "u01", "u02"]

@st.cache_data
def get_user_dates(user_id: str):
    """Load dates with valid features for the user."""
    csv_path = DATA_DIR / "features" / f"{user_id}_features.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            return df['date'].tolist()
        except Exception:
            pass
    return ["2013-03-25", "2013-03-26"]

def get_prediction(user_id: str, date: str):
    """Fetch prediction from API or calculate locally if server offline."""
    try:
        r = requests.post(f"{API_URL}/predict/{user_id}?date={date}", timeout=5.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
        
    # Local fallback prediction
    import joblib
    from src.data.preprocessor import score_to_label
    from src.evaluation.explainability import get_top_3_shap_contributors
    from src.advice.generator import generate_advice
    
    xgb_path = REGISTRY_DIR / "xgboost_model.pkl"
    iso_path = REGISTRY_DIR / "isoforest_model.pkl"
    csv_path = DATA_DIR / "features" / f"{user_id}_features.csv"
    
    if not (xgb_path.exists() and csv_path.exists()):
        return None
        
    try:
        df_user = pd.read_csv(csv_path)
        df_day = df_user[df_user['date'] == date]
        if df_day.empty:
            return None
            
        exclude_cols = ['user_id', 'date', 'sleep_score']
        feat_dict = {c: float(df_day.iloc[0][c]) for c in df_day.columns if c not in exclude_cols}
        
        xgb_model = joblib.load(xgb_path)
        df_row = pd.DataFrame([feat_dict])
        
        pred_score = float(xgb_model.predict(df_row)[0])
        pred_score = float(np.clip(pred_score, 0.0, 3.0))
        pred_label = score_to_label(pred_score)
        
        anomaly_flag = 0
        if iso_path.exists():
            iso_model = joblib.load(iso_path)
            df_anomaly = df_row.copy()
            df_anomaly['user_id'] = user_id
            df_anomaly['date'] = date
            anomaly_flag = int(iso_model.predict_anomaly_flag(df_anomaly)[0])
            
        top_features = get_top_3_shap_contributors(df_row)
        advice_list = generate_advice(top_features)
        
        return {
            'user_id': user_id,
            'date': date,
            'predicted_score': pred_score,
            'predicted_label': pred_label,
            'anomaly_flag': anomaly_flag,
            'top_features': top_features,
            'advice': advice_list
        }
    except Exception:
        return None

def get_prediction_history(user_id: str):
    """Load user history from API or fallback locally from CSV targets."""
    try:
        r = requests.get(f"{API_URL}/history/{user_id}", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
        
    # Local fallback history
    csv_path = DATA_DIR / "features" / f"{user_id}_features.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Map actual rating if sleep_score column is present
            from src.data.preprocessor import score_to_label
            history = []
            for _, row in df.iterrows():
                history.append({
                    'date': row['date'],
                    'predicted_score': 1.8, # fallback dummy
                    'predicted_label': 'Fairly good',
                    'anomaly_flag': 0,
                    'actual_rating': score_to_label(row['sleep_score']) if not pd.isna(row['sleep_score']) else None,
                    'created_at': 'N/A'
                })
            return history
        except Exception:
            pass
    return []

# ── Sidebar Configuration ──────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/color/96/sleeping-in-bed.png", width=80)
st.sidebar.title("SleepSense Control")

users = get_user_list()
selected_user = st.sidebar.selectbox("Select User Profile", users)

dates = get_user_dates(selected_user)
selected_date = st.sidebar.selectbox("Select Target Date", dates)

menu = st.sidebar.radio(
    "Navigation", 
    ["Home Dashboard", "Daily Breakdown", "Trends & Anomalies", "SHAP Explanations", "Cohort Explorer"]
)

# Trigger manual DB Ingestion
if st.sidebar.button("Ingest User Data to DB"):
    try:
        r = requests.post(f"{API_URL}/users/{selected_user}/ingest", timeout=30.0)
        if r.status_code == 200:
            st.sidebar.success(r.json()['message'])
        else:
            st.sidebar.error(r.json()['detail'])
    except Exception as e:
        st.sidebar.error(f"Could not connect to API: {str(e)}")

# ── Page Rendering ─────────────────────────────────────────────────────────────

pred_data = get_prediction(selected_user, selected_date)

if pred_data is None:
    st.error("Error generating predictions. Please make sure the preprocessed features exist.")
else:
    # ── 1. Home Dashboard ──────────────────────────────────────────────────────
    if menu == "Home Dashboard":
        st.title(f"🌙 SleepSense Dashboard — {selected_user}")
        st.write(f"Predicting tonight's sleep quality using behavior from **{selected_date}**.")
        
        # Anomaly Banner
        if pred_data['anomaly_flag'] == 1:
            st.markdown(
                "<div class='anomaly-banner'>⚠️ <b>Atypical Behavior Detected:</b> "
                "The user's behavior today deviated significantly from their usual baseline. "
                "The sleep quality prediction might be affected by anomalous behavior.</div>",
                unsafe_allow_html=True
            )
            
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Predicted Sleep Quality")
            
            # Plotly Sleep Score Gauge
            score = pred_data['predicted_score']
            label = pred_data['predicted_label']
            
            # Gradient Color Map
            gauge_color = "#3b82f6"
            if label == "Very good": gauge_color = "#10b981"
            elif label == "Fairly good": gauge_color = "#34d399"
            elif label == "Fairly bad": gauge_color = "#fbbf24"
            elif label == "Very bad": gauge_color = "#ef4444"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Sleep Rating: {label}", 'font': {'size': 20, 'color': '#ffffff'}},
                number = {'font': {'size': 48, 'color': '#ffffff'}},
                gauge = {
                    'axis': {'range': [0, 3], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "rgba(30, 41, 59, 0.5)",
                    'borderwidth': 2,
                    'bordercolor': "#6366f1",
                    'steps': [
                        {'range': [0, 0.5], 'color': 'rgba(239, 68, 68, 0.1)'},
                        {'range': [0.5, 1.5], 'color': 'rgba(251, 191, 36, 0.1)'},
                        {'range': [1.5, 2.5], 'color': 'rgba(52, 211, 153, 0.1)'},
                        {'range': [2.5, 3.0], 'color': 'rgba(16, 185, 129, 0.1)'}
                    ],
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### 💡 Actionable Sleep Advice")
            st.write("Top recommendations driven by today's behavior:")
            for adv in pred_data['advice']:
                st.markdown(f"<div class='advice-card'>{adv}</div>", unsafe_allow_html=True)
                
    # ── 2. Daily Breakdown ────────────────────────────────────────────────────
    elif menu == "Daily Breakdown":
        st.title(f"📊 Daytime Behavior Breakdown — {selected_date}")
        
        # Load user features
        csv_path = DATA_DIR / "features" / f"{selected_user}_features.csv"
        if csv_path.exists():
            df_user = pd.read_csv(csv_path)
            df_day = df_user[df_user['date'] == selected_date].iloc[0]
            df_user_mean = df_user.mean(numeric_only=True)
            
            st.write("Compare today's metrics against your historical average:")
            
            categories = {
                "Phone Lock/Unlock": ['unlock_count_late_night', 'unlock_count_evening', 'unlock_count_daytime'],
                "Activity (Minutes)": ['walking_minutes', 'running_minutes'],
                "App Use (Minutes)": ['app_social_min', 'app_entertainment_evening_min', 'app_study_sessions'],
                "Environment & Social": ['silence_ratio', 'conversation_ratio', 'stress_level']
            }
            
            col1, col2 = st.columns(2)
            idx = 0
            for name, feats in categories.items():
                target_col = col1 if idx % 2 == 0 else col2
                with target_col:
                    st.markdown(f"#### {name}")
                    compare_rows = []
                    for f in feats:
                        if f in df_day:
                            compare_rows.append({
                                'Metric': f.replace("_", " ").title(),
                                'Today': float(df_day[f]),
                                'Historical Mean': float(df_user_mean[f])
                            })
                    if compare_rows:
                        df_compare = pd.DataFrame(compare_rows)
                        fig = px.bar(
                            df_compare.melt(id_vars='Metric', value_vars=['Today', 'Historical Mean'], var_name='Type', value_name='Value'),
                            x='Metric', y='Value', color='Type', barmode='group',
                            color_discrete_map={'Today': '#6366f1', 'Historical Mean': 'rgba(99, 102, 241, 0.4)'}
                        )
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend_title=None, height=250)
                        st.plotly_chart(fig, use_container_width=True)
                idx += 1
                
    # ── 3. Trends & Anomalies ──────────────────────────────────────────────────
    elif menu == "Trends & Anomalies":
        st.title("📈 Longitudinal Trends & Anomalies")
        
        history = get_prediction_history(selected_user)
        if history:
            df_hist = pd.DataFrame(history).sort_values('date')
            
            st.markdown("### Predicted Sleep Score History")
            # Timeline chart
            fig = px.line(
                df_hist, x='date', y='predicted_score', 
                labels={'predicted_score': 'Predicted Score', 'date': 'Date'},
                markers=True, color_discrete_sequence=['#6366f1']
            )
            # Highlight anomalies
            df_anom = df_hist[df_hist['anomaly_flag'] == 1]
            if not df_anom.empty:
                fig.add_trace(go.Scatter(
                    x=df_anom['date'], y=df_anom['predicted_score'],
                    mode='markers', marker=dict(color='#ef4444', size=12, symbol='triangle-up'),
                    name='Behavior Anomaly'
                ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomalies list
            st.markdown("### 🚨 Behaviorally Anomalous Days Log")
            if not df_anom.empty:
                st.write(f"The system flagged **{len(df_anom)}** atypical behavior days:")
                st.dataframe(
                    df_anom[['date', 'predicted_label', 'predicted_score', 'actual_rating']],
                    use_container_width=True
                )
            else:
                st.success("No anomalies detected in the user's history! Sleep rhythms and behavior patterns look highly consistent.")
        else:
            st.info("No prediction history logged yet for this user. Predict some days to populate the history.")
            
    # ── 4. SHAP Explanations ──────────────────────────────────────────────────
    elif menu == "SHAP Explanations":
        st.title("🧩 Local SHAP Explanations")
        st.write("Understand which daily features pushed today's sleep quality prediction away from the model's baseline.")
        
        top_feats = pred_data['top_features']
        if top_feats:
            df_shap = pd.DataFrame(top_feats)
            # Sort by SHAP value for visualization
            df_shap = df_shap.sort_values('shap_value')
            
            df_shap['Feature Name'] = df_shap['feature'].str.replace('_', ' ').str.title()
            df_shap['Direction'] = np.where(df_shap['shap_value'] >= 0, 'Helped Sleep', 'Hurt Sleep')
            
            fig = px.bar(
                df_shap, x='shap_value', y='Feature Name', color='Direction',
                labels={'shap_value': 'SHAP Contribution Score', 'Feature Name': 'Behavioral Feature'},
                color_discrete_map={'Helped Sleep': '#10b981', 'Hurt Sleep': '#ef4444'},
                orientation='h'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                legend_title=None,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(
                "💡 **How to read SHAP:** Features extending to the **right** (green) improved sleep potential. "
                "Features extending to the **left** (red) decreased sleep potential. The length of the bar shows "
                "the magnitude of the feature's impact on tonight's sleep prediction."
            )
        else:
            st.warning("No SHAP explainability data available for this prediction.")
            
    # ── 5. Cohort Explorer ─────────────────────────────────────────────────────
    elif menu == "Cohort Explorer":
        st.title("👥 User Cohort Explorer")
        st.write("Explore cross-user distributions and general dataset metrics.")
        
        parquet_path = DATA_DIR / "merged" / "dataset.parquet"
        if parquet_path.exists():
            df_all = pd.read_parquet(parquet_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Distribution of Sleep Score across Cohort")
                fig = px.histogram(
                    df_all, x='sleep_score', 
                    color_discrete_sequence=['#818cf8'],
                    labels={'sleep_score': 'Sleep EMA Score'}
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.markdown("#### User Activity vs. Sleep Quality")
                # Group by user to compare average walking time vs average sleep score
                df_user_avg = df_all.groupby('user_id').mean(numeric_only=True).reset_index()
                fig = px.scatter(
                    df_user_avg, x='walking_minutes', y='sleep_score',
                    hover_name='user_id', color='psqi_pre_score',
                    labels={'walking_minutes': 'Avg Walk Minutes / Day', 'sleep_score': 'Avg Sleep Score'},
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Preprocessed cohort dataset not found. Run dataset build first.")
