import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Title
    cells.append(nbf.v4.new_markdown_cell(
        "# SleepSense — Model Training & Evaluation (LOUO-CV)\n\n"
        "This notebook trains and evaluates the baseline, Random Forest, and XGBoost regressor models using Leave-One-User-Out (LOUO) cross-validation. "
        "It generates a model comparison table and plots performance metrics, confusion matrices, and prediction trends."
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
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n"
        "from src.models.trainer import load_data, run_louo_cv\n\n"
        "# Set plotting style\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "plt.rcParams[\"figure.figsize\"] = (10, 6)"
    ))
    
    # Run CV
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Run Leave-One-User-Out Cross-Validation\n\n"
        "We will evaluate all three models using LOUO-CV to ensure they generalize to completely unseen users."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "df, X, y, user_ids = load_data()\n\n"
        "print(\"Evaluating Baseline Heuristic (User Mean)... \")\n"
        "y_pred_base, base_metrics = run_louo_cv('baseline', X, y, user_ids)\n\n"
        "print(\"Evaluating Random Forest Regressor... \")\n"
        "y_pred_rf, rf_metrics = run_louo_cv('rf', X, y, user_ids)\n\n"
        "print(\"Evaluating XGBoost Regressor... \")\n"
        "y_pred_xgb, xgb_metrics = run_louo_cv('xgb', X, y, user_ids)\n\n"
        "print(\"CV completed successfully.\")"
    ))
    
    # Model Comparison Table
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Model Performance Comparison\n\n"
        "Let's compare the performance of each model."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "metrics_dict = {\n"
        "    'Baseline': base_metrics,\n"
        "    'Random Forest': rf_metrics,\n"
        "    'XGBoost': xgb_metrics\n"
        "}\n\n"
        "metrics_df = pd.DataFrame(metrics_dict).T\n"
        "print(metrics_df)\n\n"
        "# Save comparison table as CSV\n"
        "os.makedirs('../implementation', exist_ok=True)\n"
        "os.makedirs('implementation', exist_ok=True)\n"
        "try:\n"
        "    metrics_df.to_csv('../implementation/model_comparison.csv')\n"
        "except Exception:\n"
        "    metrics_df.to_csv('implementation/model_comparison.csv')"
    ))
    
    # Plot Metrics
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Visualize Model Comparison"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "plot_df = metrics_df.reset_index().rename(columns={'index': 'Model'})\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "# MAE & RMSE\n"
        "plot_df_melted = plot_df.melt(id_vars='Model', value_vars=['mae', 'rmse'], var_name='Metric', value_name='Value')\n"
        "sns.barplot(data=plot_df_melted, x='Model', y='Value', hue='Metric', ax=axes[0], palette='muted')\n"
        "axes[0].set_title(\"Regression Errors (Lower is Better)\")\n"
        "axes[0].set_ylabel(\"Error Value\")\n\n"
        "# Accuracies\n"
        "plot_df_acc = plot_df.melt(id_vars='Model', value_vars=['binary_accuracy', 'ordinal_accuracy'], var_name='Metric', value_name='Accuracy')\n"
        "sns.barplot(data=plot_df_acc, x='Model', y='Accuracy', hue='Metric', ax=axes[1], palette='pastel')\n"
        "axes[1].set_title(\"Classification Accuracies (Higher is Better)\")\n"
        "axes[1].set_ylabel(\"Accuracy (%)\")\n"
        "axes[1].set_ylim(0, 1.05)\n\n"
        "plt.tight_layout()\n"
        "try:\n"
        "    plt.savefig('../implementation/model_metrics_comparison.png', dpi=150)\n"
        "except Exception:\n"
        "    plt.savefig('implementation/model_metrics_comparison.png', dpi=150)\n"
        "plt.show()"
    ))
    
    # Confusion Matrix
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Confusion Matrix (XGBoost)\n\n"
        "Let's see how our best model (XGBoost) performs across individual sleep quality labels after post-rounding."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Round predictions to integers and clip\n"
        "y_pred_rounded = np.clip(np.round(y_pred_xgb), 0, 3).astype(int)\n"
        "y_true = y.values.astype(int)\n\n"
        "cm = confusion_matrix(y_true, y_pred_rounded, labels=[0, 1, 2, 3])\n"
        "labels = ['Very bad', 'Fairly bad', 'Fairly good', 'Very good']\n\n"
        "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)\n"
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        "disp.plot(cmap='Blues', ax=ax, values_format='d')\n"
        "plt.title(\"Sleep Quality Confusion Matrix (XGBoost)\")\n"
        "plt.grid(False)\n"
        "plt.tight_layout()\n"
        "try:\n"
        "    plt.savefig('../implementation/model_confusion_matrix.png', dpi=150)\n"
        "except Exception:\n"
        "    plt.savefig('implementation/model_confusion_matrix.png', dpi=150)\n"
        "plt.show()"
    ))
    
    # Predicted vs Actual scatter
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Predicted vs. Actual Ratings\n\n"
        "Let's look at the distribution of continuous predictions against the true ordinal ratings."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "plt.figure(figsize=(8, 6))\n"
        "sns.stripplot(x=y_true, y=y_pred_xgb, jitter=0.25, size=5, alpha=0.5, palette='Set1', legend=False)\n"
        "sns.boxplot(x=y_true, y=y_pred_xgb, color='white', width=0.4, boxprops=dict(alpha=0.5))\n"
        "plt.xticks([0, 1, 2, 3], labels)\n"
        "plt.title(\"Continuous Predictions vs. Actual Sleep Ratings\")\n"
        "plt.xlabel(\"True Sleep Rating\")\n"
        "plt.ylabel(\"Continuous Predicted Score\")\n"
        "plt.tight_layout()\n"
        "try:\n"
        "    plt.savefig('../implementation/model_predictions_vs_actual.png', dpi=150)\n"
        "except Exception:\n"
        "    plt.savefig('implementation/model_predictions_vs_actual.png', dpi=150)\n"
        "plt.show()"
    ))
    
    nb['cells'] = cells
    
    with open('notebooks/03_model_training.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook 03_model_training.ipynb built successfully!")

if __name__ == '__main__':
    build_notebook()
