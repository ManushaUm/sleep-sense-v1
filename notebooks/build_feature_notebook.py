import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Title
    cells.append(nbf.v4.new_markdown_cell(
        "# SleepSense — Feature Engineering & EDA Validation\n\n"
        "This notebook validates the feature store pipeline, explores the relationships between passive sensing/survey features and sleep quality ratings, and prepares the dataset for machine learning models."
    ))
    
    # Imports
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Set plotting style\n"
        "sns.set_theme(style=\"whitegrid\")\n"
        "plt.rcParams[\"figure.figsize\"] = (10, 6)\n\n"
        "# Load the merged preprocessed dataset\n"
        "dataset_path = \"../data/preprocessed/merged/dataset.parquet\"\n"
        "if not os.path.exists(dataset_path):\n"
        "    dataset_path = \"data/preprocessed/merged/dataset.parquet\"\n\n"
        "df = pd.read_parquet(dataset_path)\n"
        "print(f\"Dataset shape: {df.shape}\")\n"
        "print(f\"Columns: {list(df.columns)}\")"
    ))
    
    # Class Balance Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Class Balance: Target Label Distribution\n\n"
        "Let's inspect the target variable `sleep_score`."
    ))
    
    # Class Balance Code
    cells.append(nbf.v4.new_code_cell(
        "label_map = {3: 'Very good', 2: 'Fairly good', 1: 'Fairly bad', 0: 'Very bad'}\n"
        "df['sleep_label'] = df['sleep_score'].map(label_map)\n\n"
        "target_counts = df['sleep_score'].value_counts().sort_index()\n"
        "target_pct = df['sleep_score'].value_counts(normalize=True).sort_index() * 100\n\n"
        "for val, count in target_counts.items():\n"
        "    print(f\"{label_map[val]} ({val}): {count} samples ({target_pct[val]:.1f}%)\")\n\n"
        "# Plot target distribution\n"
        "plt.figure(figsize=(6, 4))\n"
        "sns.countplot(data=df, x='sleep_label', order=['Very bad', 'Fairly bad', 'Fairly good', 'Very good'], palette='viridis')\n"
        "plt.title(\"Sleep Quality Rating Distribution (EMA Target)\")\n"
        "plt.xlabel(\"Sleep Quality Label\")\n"
        "plt.ylabel(\"Sample Count\")\n"
        "plt.tight_layout()\n"
        "os.makedirs('../implementation', exist_ok=True)\n"
        "os.makedirs('implementation', exist_ok=True)\n"
        "try:\n"
        "    plt.savefig('../implementation/eda_class_distribution.png', dpi=150)\n"
        "except Exception:\n"
        "    plt.savefig('implementation/eda_class_distribution.png', dpi=150)\n"
        "plt.show()"
    ))
    
    # Missing Values Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Missing Value Analysis\n\n"
        "Let's examine how much missing data is present in each column. Some modalities may have higher missingness due to phone power-offs or sensors being disabled."
    ))
    
    # Missing Values Code
    cells.append(nbf.v4.new_code_cell(
        "missing_counts = df.isnull().sum()\n"
        "missing_pct = (df.isnull().sum() / len(df)) * 100\n"
        "missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Percentage (%)': missing_pct})\n"
        "missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage (%)', ascending=False)\n\n"
        "if missing_df.empty:\n"
        "    print(\"No missing values found in the dataset! All features are complete.\")\n"
        "else:\n"
        "    print(\"Columns with missing values:\")\n"
        "    print(missing_df)\n\n"
        "    # Plot missingness\n"
        "    plt.figure(figsize=(10, 5))\n"
        "    sns.barplot(x=missing_df.index, y=missing_df['Percentage (%)'], palette='magma')\n"
        "    plt.xticks(rotation=45, ha='right')\n"
        "    plt.title(\"Missing Value Percentage per Feature\")\n"
        "    plt.ylabel(\"Missing Percentage (%)\")\n"
        "    plt.tight_layout()\n"
        "    try:\n"
        "        plt.savefig('../implementation/eda_missing_values.png', dpi=150)\n"
        "    except Exception:\n"
        "        plt.savefig('implementation/eda_missing_values.png', dpi=150)\n"
        "    plt.show()"
    ))
    
    # Correlation Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Feature Correlation with Sleep Quality Target\n\n"
        "Let's look at which daytime behavioral signals are most strongly correlated with sleep quality."
    ))
    
    # Correlation Code
    cells.append(nbf.v4.new_code_cell(
        "exclude_cols = ['user_id', 'date', 'sleep_label']\n"
        "numeric_cols = [c for c in df.columns if c not in exclude_cols]\n\n"
        "correlations = df[numeric_cols].corr()['sleep_score'].sort_values()\n"
        "print(\"Top 10 negative correlations with sleep quality:\")\n"
        "print(correlations.head(10))\n"
        "print(\"\\nTop 10 positive correlations with sleep quality:\")\n"
        "print(correlations.tail(10))\n\n"
        "# Plot top correlations\n"
        "top_corr_features = pd.concat([correlations.head(8), correlations.tail(8)]).drop('sleep_score', errors='ignore').sort_values()\n"
        "plt.figure(figsize=(10, 6))\n"
        "sns.barplot(x=top_corr_features.values, y=top_corr_features.index, palette='coolwarm')\n"
        "plt.title(\"Feature Correlations with Sleep Score (Target)\")\n"
        "plt.xlabel(\"Pearson Correlation Coefficient\")\n"
        "plt.ylabel(\"Features\")\n"
        "plt.axvline(x=0, color='black', linestyle='--', linewidth=1)\n"
        "plt.tight_layout()\n"
        "try:\n"
        "    plt.savefig('../implementation/eda_correlations.png', dpi=150)\n"
        "except Exception:\n"
        "    plt.savefig('implementation/eda_correlations.png', dpi=150)\n"
        "plt.show()"
    ))
    
    # Behavioral Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Key Behavioral Distributions Grouped by Sleep Quality\n\n"
        "Let's check the distributions of some highly relevant predictors across sleep quality categories:\n"
        "- Phone unlocks at late night (`unlock_count_late_night`)\n"
        "- Stationary time ratio (`stationary_ratio`)\n"
        "- Evening entertainment app use (`app_entertainment_evening_min`)\n"
        "- Stress levels (`stress_level`)"
    ))
    
    # Behavioral Code
    cells.append(nbf.v4.new_code_cell(
        "features_to_plot = [\n"
        "    'unlock_count_late_night',\n"
        "    'stationary_ratio',\n"
        "    'app_entertainment_evening_min',\n"
        "    'stress_level'\n"
        "]\n\n"
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
        "axes = axes.flatten()\n\n"
        "for idx, feat in enumerate(features_to_plot):\n"
        "    if feat in df.columns:\n"
        "        sns.boxplot(\n"
        "            data=df, \n"
        "            x='sleep_label', \n"
        "            y=feat, \n"
        "            ax=axes[idx],\n"
        "            order=['Very bad', 'Fairly bad', 'Fairly good', 'Very good'],\n"
        "            palette='Set2'\n"
        "        )\n"
        "        axes[idx].set_title(f\"Distribution of {feat} by Sleep Quality\")\n"
        "        axes[idx].set_xlabel(\"Sleep Quality\")\n"
        "        axes[idx].set_ylabel(feat)\n\n"
        "plt.tight_layout()\n"
        "try:\n"
        "    plt.savefig('../implementation/eda_behavior_distributions.png', dpi=150)\n"
        "except Exception:\n"
        "    plt.savefig('implementation/eda_behavior_distributions.png', dpi=150)\n"
        "plt.show()"
    ))
    
    # Variance Markdown
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Variance Analysis: Within-User vs. Between-User Variance\n\n"
        "Since we are dealing with multi-user longitudinal lifelog data, it is critical to see if feature variance is mostly driven by differences *between* individuals (e.g. some people naturally study more or walk more) or differences *within* individuals day-to-day."
    ))
    
    # Variance Code
    cells.append(nbf.v4.new_code_cell(
        "variance_records = []\n"
        "for col in numeric_cols:\n"
        "    if col == 'sleep_score':\n"
        "        continue\n"
        "    global_std = df[col].std()\n"
        "    within_std = df.groupby('user_id')[col].std().mean()\n"
        "    \n"
        "    variance_records.append({\n"
        "        'Feature': col,\n"
        "        'Global Std': global_std,\n"
        "        'Within-User Std': within_std,\n"
        "        'Ratio (Within/Global)': within_std / global_std if global_std > 0 else np.nan\n"
        "    })\n\n"
        "var_df = pd.DataFrame(variance_records).dropna().sort_values(by='Ratio (Within/Global)')\n"
        "print(\"Top 10 features with mostly between-user differences (low within-user variance ratio):\")\n"
        "print(var_df.head(10))\n"
        "print(\"\\nTop 10 features with mostly day-to-day within-user variation (high within-user variance ratio):\")\n"
        "print(var_df.tail(10))\n\n"
        "# Plot top/bottom variance ratios\n"
        "plt.figure(figsize=(10, 6))\n"
        "sns.barplot(data=var_df.head(15), x='Ratio (Within/Global)', y='Feature', palette='crest')\n"
        "plt.title(\"Variance Ratio (Within-User / Global Variance) - Predictability Indicator\")\n"
        "plt.xlabel(\"Within-User Std / Global Std Ratio\")\n"
        "plt.ylabel(\"Feature\")\n"
        "plt.axvline(x=1.0, color='red', linestyle='--', linewidth=1, label='Perfectly Equal')\n"
        "plt.tight_layout()\n"
        "try:\n"
        "    plt.savefig('../implementation/eda_variance_ratios.png', dpi=150)\n"
        "except Exception:\n"
        "    plt.savefig('implementation/eda_variance_ratios.png', dpi=150)\n"
        "plt.show()"
    ))
    
    nb['cells'] = cells
    
    with open('notebooks/02_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook 02_feature_engineering.ipynb built successfully!")

if __name__ == '__main__':
    build_notebook()
