"""
Simple Data Processing - Creates clean numeric dataset for training
"""

import pandas as pd
import numpy as np
import os

print("="*60)
print("SIMPLE DATA PROCESSING")
print("="*60)

# Find the data file
data_dir = 'data'
files = os.listdir(data_dir)
print(f"\nFiles in data folder: {files}")

# Find the original Excel file (not the one with target)
excel_file = None
for f in files:
    if f.endswith('.xlsx') and 'target' not in f.lower():
        excel_file = f
        break

if excel_file is None:
    # If no original, use any excel file
    for f in files:
        if f.endswith('.xlsx'):
            excel_file = f
            break

if excel_file is None:
    print("ERROR: No Excel file found!")
    exit(1)

print(f"\nUsing data file: {excel_file}")

# Load data
file_path = os.path.join(data_dir, excel_file)
df = pd.read_excel(file_path)
print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}")

# Create target variable using simple logic
print("\n" + "="*60)
print("CREATING TARGET VARIABLE")
print("="*60)

# Make sure date column is proper datetime
if 'TransactionStartTime' in df.columns:
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    # Drop rows with invalid dates
    df = df.dropna(subset=['TransactionStartTime'])
    print(f"After dropping invalid dates: {len(df):,} rows")

# Calculate recency (days since last transaction per customer)
snapshot = df['TransactionStartTime'].max()
print(f"Snapshot date: {snapshot}")

# Group by customer
customer_data = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot - x.max()).days,
    'TransactionId': 'count',
    'Amount': 'sum'
}).rename(columns={
    'TransactionStartTime': 'recency',
    'TransactionId': 'frequency',
    'Amount': 'monetary'
}).reset_index()

# Handle any NaN
customer_data = customer_data.fillna(0)

# Simple risk scoring: high recency + low frequency = high risk
customer_data['recency_norm'] = (customer_data['recency'] - customer_data['recency'].min()) / (customer_data['recency'].max() - customer_data['recency'].min() + 1)
customer_data['frequency_norm'] = 1 - (customer_data['frequency'] - customer_data['frequency'].min()) / (customer_data['frequency'].max() - customer_data['frequency'].min() + 1)
customer_data['monetary_norm'] = 1 - (customer_data['monetary'] - customer_data['monetary'].min()) / (customer_data['monetary'].max() - customer_data['monetary'].min() + 1)

# Combined risk score
customer_data['risk_score'] = (customer_data['recency_norm'] + customer_data['frequency_norm'] + customer_data['monetary_norm']) / 3

# Label top 20% as high risk
threshold = customer_data['risk_score'].quantile(0.8)
customer_data['is_high_risk'] = (customer_data['risk_score'] >= threshold).astype(int)

print(f"\nHigh risk customers: {customer_data['is_high_risk'].sum():,} ({customer_data['is_high_risk'].mean()*100:.1f}%)")
print(f"Low risk customers: {(customer_data['is_high_risk']==0).sum():,} ({100 - customer_data['is_high_risk'].mean()*100:.1f}%)")

# Merge back to original data
print("\n" + "="*60)
print("CREATING FINAL DATASET")
print("="*60)

df_final = df.merge(customer_data[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
df_final['is_high_risk'] = df_final['is_high_risk'].fillna(0)

# Drop unnecessary columns
drop_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'TransactionStartTime']
for col in drop_cols:
    if col in df_final.columns:
        df_final = df_final.drop(columns=[col])

# Convert categorical to numeric using one-hot encoding
cat_cols = df_final.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {cat_cols}")

for col in cat_cols:
    if col != 'CustomerId':
        dummies = pd.get_dummies(df_final[col], prefix=col, drop_first=True)
        df_final = pd.concat([df_final, dummies], axis=1)
        df_final = df_final.drop(columns=[col])

# Drop CustomerId
if 'CustomerId' in df_final.columns:
    df_final = df_final.drop(columns=['CustomerId'])

# Ensure all numeric
for col in df_final.columns:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)

# Save
output_path = os.path.join(data_dir, 'processed_data.csv')
df_final.to_csv(output_path, index=False)

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Features: {len(df_final.columns) - 1}")
print(f"Target distribution:")
print(df_final['is_high_risk'].value_counts())

print("\n" + "="*60)
print("✅ Processing complete!")
print(f"Saved to: {output_path}")
print("="*60)
