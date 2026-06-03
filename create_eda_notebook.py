# create_eda_notebook.py
import json

notebook_content = {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Settings\n",
    "plt.style.use('seaborn-v0_8-darkgrid')\n",
    "sns.set_palette(\"husl\")\n",
    "pd.set_option('display.max_columns', None)\n",
    "\n",
    "print(\"Libraries loaded!\") - create_eda_notebook.py:25"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load your Excel data\n",
    "df = pd.read_excel('data/data.xlsx')\n",
    "\n",
    "print(f\"Data loaded: {df.shape}\")\n - create_eda_notebook.py:37",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data overview\n",
    "print(\"Data Types:\")\n - create_eda_notebook.py:48",
    "print(df.dtypes)\n - create_eda_notebook.py:49",
    "print(\"\\n\" + \"=\"*50)\n - create_eda_notebook.py:50",
    "print(\"Dataset Info:\")\n - create_eda_notebook.py:51",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Summary statistics\n",
    "num_cols = df.select_dtypes(include=[np.number]).columns\n",
    "print(\"Summary Statistics:\")\n - create_eda_notebook.py:63",
    "df[num_cols].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for missing values\n",
    "print(\"Missing Values:\")\n - create_eda_notebook.py:74",
    "print(df.isnull().sum())\n - create_eda_notebook.py:75",
    "print(f\"\\nTotal missing: {df.isnull().sum().sum()}\") - create_eda_notebook.py:76"
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

# Save the notebook
import os
os.makedirs('notebooks', exist_ok=True)

with open('notebooks/eda.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print("✅ Notebook created at: notebooks/eda.ipynb - create_eda_notebook.py:110")
print("📓 Open with: jupyter notebook notebooks/eda.ipynb - create_eda_notebook.py:111")