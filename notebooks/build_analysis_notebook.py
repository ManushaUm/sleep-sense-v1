import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Title
    cells.append(nbf.v4.new_markdown_cell(
        "# SleepSense — SHAP Explainability & Anomaly Analysis\n\n"
        "This notebook performs post-hoc model explainability using SHAP values "
        "and analyzes the daily behavioral anomalies flagged by our Isolation Forest models."
    ))
    
    # Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import sys\n"
        "sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import joblib\n"
        "import shap\n"
        "from src.models.trainer import load_data\n"
        "from src.evaluation.explainability import explain_dataset, plot_shap_summary, load_xgb_pipeline\n\n"
        "# Set plotting style\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "plt.rcParams[\"figure.figsize\"] = (10, 6)"
    ))
    
    # Run SHAP
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. SHAP Global Feature Importance\n\n"
        "Let's compute SHAP values for all daily behavioral predictions to see what drivers most heavily impact our sleep predictions globally."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "df, X, y, user_ids = load_data()\n\n"
        "print(\"Computing SHAP values... \")\n"
        "explainer, shap_values, X_preprocessed = explain_dataset(X, save_importance=True)\n"
        "print(\"SHAP values computed and global feature importance saved successfully.\")"
    ))
    
    # Plot SHAP Summary
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Visualize SHAP Global Importances\n\n"
        "Let's display the SHAP summary plots (beeswarm and bar chart) to analyze the positive/negative impact direction of features."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Save beeswarm summary plot\n"
        "os.makedirs('../implementation', exist_ok=True)\n"
        "os.makedirs('implementation', exist_ok=True)\n"
        "shap_summary_path = '../implementation/shap_summary_beeswarm.png'\n"
        "try:\n"
        "    plot_shap_summary(shap_values, X_preprocessed, shap_summary_path)\n"
        "except Exception:\n"
        "    plot_shap_summary(shap_values, X_preprocessed, 'implementation/shap_summary_beeswarm.png')\n\n"
        "# Plot SHAP global bar chart in notebook\n"
        "plt.figure(figsize=(10, 6))\n"
        "shap.plots.bar(shap_values, show=False)\n"
        "plt.title(\"SHAP Global Feature Importance (Average Absolute Impact)\")\n"
        "plt.tight_layout()\n"
        "try:\n"
        "    plt.savefig('../implementation/shap_importance_bar.png', dpi=150)\n"
        "except Exception:\n"
        "    plt.savefig('implementation/shap_importance_bar.png', dpi=150)\n"
        "plt.show()"
    ))
    
    # Anomaly Detection
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Behavioral Anomaly Detection\n\n"
        "Now let's load our trained per-user Isolation Forest model and inspect behavioral anomalies across days."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "iso_model_path = '../models/registry/isoforest_model.pkl'\n"
        "if not os.path.exists(iso_model_path):\n"
        "    iso_model_path = 'models/registry/isoforest_model.pkl'\n\n"
        "user_iso = joblib.load(iso_model_path)\n\n"
        "# Compile df with user_id and date for prediction\n"
        "df_anomaly = X.copy()\n"
        "df_anomaly['user_id'] = user_ids\n\n"
        "anomaly_scores = user_iso.predict_anomaly_score(df_anomaly)\n"
        "anomaly_flags = user_iso.predict_anomaly_flag(df_anomaly)\n\n"
        "df['anomaly_score'] = anomaly_scores\n"
        "df['anomaly_flag'] = anomaly_flags\n\n"
        "total_anomalies = df['anomaly_flag'].sum()\n"
        "print(f\"Total behavior anomalies flagged: {total_anomalies} out of {len(df)} days ({total_anomalies/len(df)*100:.1f}%)\")"
    ))
    
    # Plot anomaly scores
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Visualize Anomaly Score Distribution\n\n"
        "Let's visualize the distribution of anomaly scores. Scores closer to negative values represent highly atypical behavioral days for a user."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(8, 5))\n"
        "sns.histplot(data=df, x='anomaly_score', hue='anomaly_flag', bins=30, kde=True, palette='Set1', multiple='stack')\n"
        "plt.title(\"Distribution of Behavioral Anomaly Scores\")\n"
        "plt.xlabel(\"Anomaly Score (Lower = More Anomalous)\")\n"
        "plt.ylabel(\"Count\")\n"
        "plt.axvline(x=df[df['anomaly_flag'] == 1]['anomaly_score'].max(), color='red', linestyle='--', label='Anomaly Threshold')\n"
        "plt.legend()\n"
        "plt.tight_layout()\n"
        "try:\n"
        "    plt.savefig('../implementation/anomaly_score_distribution.png', dpi=150)\n"
        "except Exception:\n"
        "    plt.savefig('implementation/anomaly_score_distribution.png', dpi=150)\n"
        "plt.show()"
    ))
    
    # Plot local anomaly example
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Local Anomaly Example: Analyzing a Flagged Day\n\n"
        "Let's look at one specific anomalous day for a user and compare their behavior on that day against their historical average."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Find a user with at least one anomaly\n"
        "anomalous_users = df[df['anomaly_flag'] == 1]['user_id'].unique()\n"
        "if len(anomalous_users) > 0:\n"
        "    selected_user = anomalous_users[0]\n"
        "    user_days = df[df['user_id'] == selected_user]\n"
        "    \n"
        "    # Get the anomalous day and normal days\n"
        "    anom_day = user_days[user_days['anomaly_flag'] == 1].iloc[0]\n"
        "    normal_days_mean = user_days[user_days['anomaly_flag'] == 0].mean(numeric_only=True)\n"
        "    \n"
        "    # Compare key features\n"
        "    compare_feats = ['unlock_count_late_night', 'stationary_ratio', 'walking_minutes', 'app_entertainment_evening_min', 'stress_level']\n"
        "    comparison_df = pd.DataFrame({\n"
        "        'Feature': compare_feats,\n"
        "        'Anomalous Day Value': [anom_day[f] for f in compare_feats],\n"
        "        'Historical Mean': [normal_days_mean[f] for f in compare_feats]\n"
        "    })\n"
        "    print(f\"Behavior comparison for user {selected_user} on anomalous day ({anom_day['date']}):\")\n"
        "    print(comparison_df)\n"
        "else:\n"
        "    print(\"No anomalous users found.\")"
    ))
    
    nb['cells'] = cells
    
    with open('notebooks/04_results_analysis.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook 04_results_analysis.ipynb built successfully!")

if __name__ == '__main__':
    build_notebook()
