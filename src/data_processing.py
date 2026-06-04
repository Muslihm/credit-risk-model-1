"""
Credit Risk Data Processing Pipeline - SIMPLE WORKING VERSION
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Get the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

print("="*60)
print("COMPLETE DATA PROCESSING PIPELINE")
print("="*60)
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")

# Find the Excel file
excel_files = []
if os.path.exists(DATA_DIR):
    for file in os.listdir(DATA_DIR):
        if file.endswith(('.xlsx', '.xls')):
            excel_files.append(file)
            print(f"Found Excel file: {file}")

if not excel_files:
    print(f"\nERROR: No Excel files found in {DATA_DIR}")
    print("Please make sure data.xlsx is in the data folder")
    sys.exit(1)

data_file = os.path.join(DATA_DIR, excel_files[0])
print(f"\nLoading data from: {data_file}")

# Load data
df = pd.read_excel(data_file)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# Create target variable using RFM
print("\n" + "="*60)
print("CREATING TARGET VARIABLE (is_high_risk)")
print("="*60)

# Convert date column
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
snapshot_date = df['TransactionStartTime'].max()
print(f"Snapshot date: {snapshot_date}")

# Calculate RFM per customer
rfm = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
    'CustomerId': 'count',
    'Amount': 'sum'
}).rename(columns={
    'TransactionStartTime': 'recency_days',
    'CustomerId': 'frequency',
    'Amount': 'monetary'
}).reset_index()

rfm['recency_days'] = rfm['recency_days'].fillna(0)
rfm['frequency'] = rfm['frequency'].fillna(0)
rfm['monetary'] = rfm['monetary'].fillna(0)

print(f"Total customers: {len(rfm):,}")

# Cap outliers
for col in ['recency_days', 'frequency', 'monetary']:
    cap = rfm[col].quantile(0.99)
    rfm[col] = rfm[col].clip(upper=cap)

# Log transform
rfm['frequency_log'] = np.log1p(rfm['frequency'])
rfm['monetary_log'] = np.log1p(rfm['monetary'])

# Scale and cluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
cluster_features = ['recency_days', 'frequency_log', 'monetary_log']
rfm_scaled = scaler.fit_transform(rfm[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

# Identify high-risk cluster
cluster_summary = rfm.groupby('cluster').agg({
    'recency_days': 'mean',
    'frequency': 'mean',
    'monetary': 'mean'
})

recency_norm = (cluster_summary['recency_days'] - cluster_summary['recency_days'].min()) / \
               (cluster_summary['recency_days'].max() - cluster_summary['recency_days'].min() + 1e-10)
frequency_norm = (cluster_summary['frequency'].max() - cluster_summary['frequency']) / \
                 (cluster_summary['frequency'].max() - cluster_summary['frequency'].min() + 1e-10)
monetary_norm = (cluster_summary['monetary'].max() - cluster_summary['monetary']) / \
                (cluster_summary['monetary'].max() - cluster_summary['monetary'].min() + 1e-10)

risk_score = recency_norm + frequency_norm + monetary_norm
high_risk_cluster = risk_score.idxmax()

rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

high_risk_count = rfm['is_high_risk'].sum()
high_risk_pct = (high_risk_count / len(rfm)) * 100
low_risk_count = len(rfm) - high_risk_count
low_risk_pct = 100 - high_risk_pct

print(f"\nHigh-risk customers: {high_risk_count:,} ({high_risk_pct:.1f}%)")
print(f"Low-risk customers: {low_risk_count:,} ({low_risk_pct:.1f}%)")

# Merge target back
print("\n" + "="*60)
print("PREPARING FINAL DATASET")
print("="*60)

df_with_target = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
df_with_target['is_high_risk'] = df_with_target['is_high_risk'].fillna(0)

# Drop unnecessary columns
cols_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'TransactionStartTime']
for col in cols_to_drop:
    if col in df_with_target.columns:
        df_with_target = df_with_target.drop(columns=[col])

# One-hot encode categorical columns
categorical_cols = df_with_target.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")

for col in categorical_cols:
    if col != 'CustomerId':
        dummies = pd.get_dummies(df_with_target[col], prefix=col, drop_first=True)
        df_with_target = pd.concat([df_with_target, dummies], axis=1)
        df_with_target = df_with_target.drop(columns=[col])

# Drop CustomerId
if 'CustomerId' in df_with_target.columns:
    df_with_target = df_with_target.drop(columns=['CustomerId'])

# Ensure all columns are numeric
for col in df_with_target.columns:
    if df_with_target[col].dtype == 'object':
        print(f"Converting column {col} to numeric")
        df_with_target[col] = pd.to_numeric(df_with_target[col], errors='coerce').fillna(0)

# Fill any NaN with 0
df_with_target = df_with_target.fillna(0)

# Save to CSV
output_path = os.path.join(DATA_DIR, 'processed_data.csv')
df_with_target.to_csv(output_path, index=False)

print(f"\nFinal dataset saved to: {output_path}")
print(f"Shape: {df_with_target.shape}")
print(f"Columns: {len(df_with_target.columns)}")
print(f"\nTarget distribution:")
target_0 = (df_with_target['is_high_risk'] == 0).sum()
target_1 = (df_with_target['is_high_risk'] == 1).sum()
pct_0 = (target_0 / len(df_with_target)) * 100
pct_1 = (target_1 / len(df_with_target)) * 100
print(f"  is_high_risk = 0: {target_0:,} ({pct_0:.1f}%)")
print(f"  is_high_risk = 1: {target_1:,} ({pct_1:.1f}%)")

print("\n" + "="*60)
print("DATA PROCESSING COMPLETE!")
print("="*60)
