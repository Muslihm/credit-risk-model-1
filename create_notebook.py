import json

# Complete notebook content as proper JSON
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Credit Risk Model - Exploratory Data Analysis (EDA)\n",
                "## Task 2: Understanding the Dataset\n",
                "\n",
                "**Objective:** Explore the dataset to uncover patterns, identify data quality issues, and form hypotheses to guide feature engineering."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import required libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set visualization style\n",
                "plt.style.use('seaborn-v0_8-darkgrid')\n",
                "sns.set_palette(\"husl\")\n",
                "\n",
                "# Display settings\n",
                "pd.set_option('display.max_columns', None)\n",
                "pd.set_option('display.max_rows', 100)\n",
                "pd.set_option('display.float_format', lambda x: '%.3f' % x)\n",
                "\n",
                "print(\"Libraries imported successfully!\") - create_notebook.py:39"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Overview of the Data\n",
                "\n",
                "### Load and understand the dataset structure"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create synthetic credit dataset\n",
                "np.random.seed(42)\n",
                "n_samples = 10000\n",
                "\n",
                "data = {\n",
                "    'credit_amount': np.random.gamma(shape=2, scale=2500, size=n_samples),\n",
                "    'age': np.random.normal(35, 12, n_samples),\n",
                "    'duration_months': np.random.exponential(24, n_samples),\n",
                "    'installment_rate': np.random.uniform(1, 4, n_samples),\n",
                "    'present_employment_years': np.random.exponential(5, n_samples),\n",
                "    'num_credits': np.random.poisson(2, n_samples),\n",
                "    'num_dependents': np.random.poisson(1, n_samples),\n",
                "    'savings_account': np.random.choice(['unknown', 'lt_100', '100_500', '500_1000', 'gt_1000'], n_samples),\n",
                "    'housing': np.random.choice(['own', 'rent', 'free'], n_samples, p=[0.6, 0.3, 0.1]),\n",
                "    'purpose': np.random.choice(['car', 'furniture', 'education', 'business', 'vacation'], n_samples),\n",
                "    'credit_history': np.random.choice(['critical', 'poor', 'good', 'very_good', 'excellent'], n_samples),\n",
                "    'job_type': np.random.choice(['unskilled', 'skilled', 'management', 'unemployed'], n_samples),\n",
                "}\n",
                "\n",
                "df = pd.DataFrame(data)\n",
                "\n",
                "# Create default probability based on risk factors\n",
                "risk_score = (\n",
                "    (df['credit_amount'] > 5000) * 0.3 +\n",
                "    (df['duration_months'] > 36) * 0.2 +\n",
                "    (df['installment_rate'] > 3) * 0.15 +\n",
                "    (df['credit_history'] == 'critical') * 0.25 +\n",
                "    (df['present_employment_years'] < 1) * 0.1\n",
                ")\n",
                "df['default'] = (np.random.random(n_samples) < risk_score).astype(int)\n",
                "\n",
                "print(\"Dataset created successfully!\\n\")\n - create_notebook.py:88",
                "print(f\"Shape: {df.shape}\")\n - create_notebook.py:89",
                "print(f\"Rows: {df.shape[0]:,}\")\n - create_notebook.py:90",
                "print(f\"Columns: {df.shape[1]}\")\n - create_notebook.py:91",
                "print(f\"Default rate: {df['default'].mean():.2%}\") - create_notebook.py:92"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display first few rows\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Data types\n",
                "df.info()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Separate numerical and categorical columns\n",
                "numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n",
                "categorical_cols = df.select_dtypes(include=['object']).columns.tolist()\n",
                "\n",
                "print(f\"Numerical columns: {numerical_cols}\")\n - create_notebook.py:125",
                "print(f\"Categorical columns: {categorical_cols}\") - create_notebook.py:126"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Summary Statistics"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Summary statistics for numerical features\n",
                "df[numerical_cols].describe()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check skewness\n",
                "skewness_stats = []\n",
                "for col in numerical_cols:\n",
                "    skewness = df[col].skew()\n",
                "    if abs(skewness) < 0.5:\n",
                "        skew_type = \"Symmetric\"\n",
                "    elif abs(skewness) < 1:\n",
                "        skew_type = \"Moderately skewed\"\n",
                "    else:\n",
                "        skew_type = \"Highly skewed\"\n",
                "    skewness_stats.append({'Feature': col, 'Skewness': skewness, 'Distribution': skew_type})\n",
                "\n",
                "pd.DataFrame(skewness_stats)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Distribution of Numerical Features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Histograms\n",
                "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n",
                "axes = axes.ravel()\n",
                "\n",
                "for idx, col in enumerate(numerical_cols):\n",
                "    axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')\n",
                "    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')\n",
                "    axes[idx].set_xlabel(col)\n",
                "    axes[idx].set_ylabel('Frequency')\n",
                "    skewness = df[col].skew()\n",
                "    axes[idx].text(0.95, 0.95, f'Skewness: {skewness:.2f}', \n",
                "                   transform=axes[idx].transAxes, ha='right', va='top',\n",
                "                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Density plots by default status\n",
                "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n",
                "axes = axes.ravel()\n",
                "\n",
                "for idx, col in enumerate(numerical_cols):\n",
                "    df[df['default']==0][col].plot(kind='density', ax=axes[idx], label='Non-Default', color='green', alpha=0.6)\n",
                "    df[df['default']==1][col].plot(kind='density', ax=axes[idx], label='Default', color='red', alpha=0.6)\n",
                "    axes[idx].set_title(f'{col} by Default Status')\n",
                "    axes[idx].legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Distribution of Categorical Features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Bar plots for categorical features\n",
                "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
                "axes = axes.ravel()\n",
                "\n",
                "for idx, col in enumerate(categorical_cols):\n",
                "    df[col].value_counts().plot(kind='bar', ax=axes[idx], color='skyblue', edgecolor='black')\n",
                "    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')\n",
                "    axes[idx].tick_params(axis='x', rotation=45)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Default rates by categorical features\n",
                "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
                "axes = axes.ravel()\n",
                "\n",
                "for idx, col in enumerate(categorical_cols):\n",
                "    default_rate = df.groupby(col)['default'].mean().sort_values(ascending=False)\n",
                "    default_rate.plot(kind='bar', ax=axes[idx], color='coral', edgecolor='black')\n",
                "    axes[idx].set_title(f'Default Rate by {col}', fontsize=12, fontweight='bold')\n",
                "    axes[idx].tick_params(axis='x', rotation=45)\n",
                "    axes[idx].set_ylabel('Default Rate')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Correlation Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Correlation matrix\n",
                "corr_matrix = df[numerical_cols].corr()\n",
                "\n",
                "plt.figure(figsize=(10, 8))\n",
                "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, \n",
                "            square=True, linewidths=1, fmt='.3f')\n",
                "plt.title('Correlation Matrix', fontsize=14, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Correlations with default\n",
                "corr_with_default = corr_matrix['default'].sort_values(ascending=False)\n",
                "print(\"Correlations with Default:\")\n - create_notebook.py:297",
                "for feature, corr in corr_with_default.items():\n",
                "    if feature != 'default':\n",
                "print(f\"  {feature}: {corr:+.4f}\") - create_notebook.py:300"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Identifying Missing Values"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check for missing values\n",
                "missing_df = pd.DataFrame({\n",
                "    'Column': df.columns,\n",
                "    'Missing_Count': df.isnull().sum(),\n",
                "    'Missing_Percentage': (df.isnull().sum() / len(df)) * 100\n",
                "})\n",
                "\n",
                "if missing_df['Missing_Count'].sum() == 0:\n",
                "print(\"✅ No missing values found in the dataset!\")\n - create_notebook.py:324",
                "else:\n",
                "print(\"⚠️ Missing values detected:\")\n - create_notebook.py:326",
                "print(missing_df[missing_df['Missing_Count'] > 0]) - create_notebook.py:327"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Outlier Detection"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Box plots for outlier detection\n",
                "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n",
                "axes = axes.ravel()\n",
                "\n",
                "outlier_summary = []\n",
                "\n",
                "for idx, col in enumerate(numerical_cols):\n",
                "    axes[idx].boxplot(df[col])\n",
                "    axes[idx].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')\n",
                "    axes[idx].set_ylabel(col)\n",
                "    \n",
                "    # Calculate outliers\n",
                "    Q1 = df[col].quantile(0.25)\n",
                "    Q3 = df[col].quantile(0.75)\n",
                "    IQR = Q3 - Q1\n",
                "    outlier_count = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])\n",
                "    outlier_pct = (outlier_count / len(df)) * 100\n",
                "    \n",
                "    axes[idx].text(0.95, 0.95, f'Outliers: {outlier_count}\\n({outlier_pct:.1f}%)', \n",
                "                   transform=axes[idx].transAxes, ha='right', va='top',\n",
                "                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Summary and Key Findings"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"=\"*60)\n - create_notebook.py:382",
                "print(\"CREDIT RISK EDA  SUMMARY\")\n - create_notebook.py:383",
                "print(\"=\"*60)\n - create_notebook.py:384",
                "\n",
                "print(\"\\n1. DATASET OVERVIEW:\")\n - create_notebook.py:386",
                "print(f\"    Total samples: {len(df):,}\")\n - create_notebook.py:387",
                "print(f\"    Features: {len(df.columns)}\")\n - create_notebook.py:388",
                "print(f\"    Default rate: {df['default'].mean():.2%}\")\n - create_notebook.py:389",
                "\n",
                "print(\"\\n2. DATA QUALITY:\")\n - create_notebook.py:391",
                "print(\"   ✅ No missing values\")\n - create_notebook.py:392",
                "\n",
                "print(\"\\n3. KEY CORRELATIONS WITH DEFAULT:\")\n - create_notebook.py:394",
                "for feature, corr in list(corr_with_default.items())[1:4]:\n",
                "print(f\"    {feature}: {corr:+.3f}\")\n - create_notebook.py:396",
                "\n",
                "print(\"\\n4. RECOMMENDATIONS:\")\n - create_notebook.py:398",
                "print(\"    Apply log transformation to credit_amount and duration_months\")\n - create_notebook.py:399",
                "print(\"    Use WoE encoding for categorical variables\")\n - create_notebook.py:400",
                "print(\"    Consider Winsorization for outliers in credit_amount\")\n - create_notebook.py:401",
                "print(\"\\n✅ Task 2  EDA Complete!\") - create_notebook.py:402"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write to file
with open('notebooks/eda.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Notebook created successfully at notebooks/eda.ipynb - create_notebook.py:433")
print("📓 You can now open it with: jupyter notebook notebooks/eda.ipynb - create_notebook.py:434")