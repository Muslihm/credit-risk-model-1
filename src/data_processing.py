
"""
Credit Risk Data Processing Pipeline
Includes feature engineering AND target variable creation
Produces model-ready dataset with is_high_risk target column
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# ============================================
# PART 1: FEATURE ENGINEERING
# ============================================

class ExtractTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        if self.time_col in df.columns:
            df[self.time_col] = pd.to_datetime(df[self.time_col])
            df['hour'] = df[self.time_col].dt.hour
            df['day_of_month'] = df[self.time_col].dt.day
            df['month'] = df[self.time_col].dt.month
            df['year'] = df[self.time_col].dt.year
            df['day_of_week'] = df[self.time_col].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_business_hour'] = df['hour'].between(9, 17).astype(int)
            df = df.drop(columns=[self.time_col])
        return df


class HandleMissingValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        if len(self.num_cols) > 0:
            self.num_imputer = SimpleImputer(strategy='median')
            self.num_imputer.fit(X[self.num_cols])
        
        if len(self.cat_cols) > 0:
            self.cat_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')
            self.cat_imputer.fit(X[self.cat_cols])
        
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        if len(self.num_cols) > 0:
            df[self.num_cols] = self.num_imputer.transform(df[self.num_cols])
        if len(self.cat_cols) > 0:
            df[self.cat_cols] = self.cat_imputer.transform(df[self.cat_cols])
        return df


class EncodeCategorical(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
        self.cat_cols = [c for c in self.cat_cols if c not in id_cols]
        self.cat_cols = [c for c in self.cat_cols if X[c].nunique() <= 20]
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        if len(self.cat_cols) > 0:
            encoded = pd.get_dummies(df[self.cat_cols], drop_first=True)
            df = df.drop(columns=self.cat_cols)
            df = pd.concat([df, encoded], axis=1)
        return df


class ScaleFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(self.num_cols) > 0:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.num_cols])
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        if len(self.num_cols) > 0:
            df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df


class DropIdColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId']
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        cols_to_drop = [c for c in self.id_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df


def create_feature_pipeline():
    """Create feature engineering pipeline"""
    return Pipeline([
        ('drop_ids', DropIdColumns()),
        ('time_features', ExtractTimeFeatures()),
        ('handle_missing', HandleMissingValues()),
        ('encode_categorical', EncodeCategorical()),
        ('scale_features', ScaleFeatures())
    ])


# ============================================
# PART 2: TARGET VARIABLE ENGINEERING (RFM + Clustering)
# ============================================

class RiskTargetEngineer:
    """Creates proxy target variable using RFM analysis and K-Means clustering"""
    
    def __init__(self, customer_id_col='CustomerId',
                 transaction_date_col='TransactionStartTime',
                 amount_col='Amount',
                 random_state=42):
        self.customer_id_col = customer_id_col
        self.transaction_date_col = transaction_date_col
        self.amount_col = amount_col
        self.random_state = random_state
        self.scaler = None
        self.kmeans = None
        
    def calculate_rfm(self, df):
        df = df.copy()
        df[self.transaction_date_col] = pd.to_datetime(df[self.transaction_date_col])
        snapshot_date = df[self.transaction_date_col].max()
        
        rfm = df.groupby(self.customer_id_col).agg({
            self.transaction_date_col: lambda x: (snapshot_date - x.max()).days,
            self.customer_id_col: 'count',
            self.amount_col: 'sum'
        }).rename(columns={
            self.transaction_date_col: 'recency_days',
            self.customer_id_col: 'frequency',
            self.amount_col: 'monetary'
        }).reset_index()
        
        rfm['recency_days'] = rfm['recency_days'].fillna(0)
        rfm['frequency'] = rfm['frequency'].fillna(0)
        rfm['monetary'] = rfm['monetary'].fillna(0)
        
        return rfm
    
    def prepare_for_clustering(self, rfm_df):
        rfm_prepared = rfm_df.copy()
        rfm_prepared = rfm_prepared.replace([np.inf, -np.inf], 0)
        rfm_prepared = rfm_prepared.dropna()
        
        for col in ['recency_days', 'frequency', 'monetary']:
            cap = rfm_prepared[col].quantile(0.99)
            rfm_prepared[col] = rfm_prepared[col].clip(upper=cap)
        
        rfm_prepared['frequency_log'] = np.log1p(rfm_prepared['frequency'])
        rfm_prepared['monetary_log'] = np.log1p(rfm_prepared['monetary'])
        
        cluster_features = ['recency_days', 'frequency_log', 'monetary_log']
        
        for col in cluster_features:
            rfm_prepared[col] = rfm_prepared[col].fillna(0)
        
        self.scaler = StandardScaler()
        rfm_scaled = self.scaler.fit_transform(rfm_prepared[cluster_features])
        rfm_scaled = np.nan_to_num(rfm_scaled, nan=0.0)
        
        return rfm_prepared, rfm_scaled
    
    def cluster_and_label(self, rfm_prepared, rfm_scaled):
        self.kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(rfm_scaled)
        rfm_prepared['cluster'] = clusters
        
        cluster_summary = rfm_prepared.groupby('cluster').agg({
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
        
        rfm_prepared['is_high_risk'] = (rfm_prepared['cluster'] == high_risk_cluster).astype(int)
        
        return rfm_prepared[['CustomerId', 'is_high_risk']]
    
    def fit_transform(self, df):
        rfm = self.calculate_rfm(df)
        rfm_prepared, rfm_scaled = self.prepare_for_clustering(rfm)
        result = self.cluster_and_label(rfm_prepared, rfm_scaled)
        return result


# ============================================
# PART 3: COMPLETE DATA PROCESSING
# ============================================

def process_data_with_target():
    """Complete data processing pipeline"""
    print("="*60)
    print("COMPLETE DATA PROCESSING PIPELINE")
    print("="*60)
    
    # Try multiple paths to find data
    possible_paths = [
        'data/data.xlsx',
        '../data/data.xlsx',
        'C:/Users/MUSLIMAH/Desktop/credit-risk-model-1/data/data.xlsx'
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"\n1. Loading raw data from: {path}")
            df = pd.read_excel(path)
            break
    
    if df is None:
        print("Error: Could not find data.xlsx")
        print("Checked paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return None
    
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Create target variable
    print("\n2. Creating proxy target variable (is_high_risk)...")
    target_engineer = RiskTargetEngineer(random_state=42)
    target_labels = target_engineer.fit_transform(df)
    print(f"   High-risk customers: {target_labels['is_high_risk'].sum():,} ({target_labels['is_high_risk'].mean()*100:.1f}%)")
    
    # Add target to dataset
    print("\n3. Adding target column to dataset...")
    df_with_target = df.merge(target_labels, on='CustomerId', how='left')
    df_with_target['is_high_risk'] = df_with_target['is_high_risk'].fillna(0)
    
    # Apply feature engineering
    print("\n4. Applying feature engineering...")
    feature_pipeline = create_feature_pipeline()
    
    customer_ids = df_with_target['CustomerId']
    X = df_with_target.drop(columns=['CustomerId'])
    X_processed = feature_pipeline.fit_transform(X)
    X_processed['CustomerId'] = customer_ids.values
    X_processed['is_high_risk'] = df_with_target['is_high_risk'].values
    
    feature_count = len(X_processed.columns) - 2
    print(f"\n5. Final processed dataset shape: {X_processed.shape}")
    print(f"   Features: {feature_count} (excluding CustomerId and is_high_risk)")
    print(f"   Target column: is_high_risk")
    
    # Save results
    output_path = 'data/processed_data.csv'
    os.makedirs('data', exist_ok=True)
    X_processed.to_csv(output_path, index=False)
    print(f"\n   Saved to: {output_path}")
    
    print("\n" + "="*60)
    print("COMPLETE! Model-ready dataset with is_high_risk target column")
    print("="*60)
    
    return X_processed


if __name__ == "__main__":
    df_processed = process_data_with_target()
    if df_processed is not None:
        print(f"\nFinal dataset columns: {df_processed.columns.tolist()[:5]}...")
        print(f"Target distribution:\n{df_processed['is_high_risk'].value_counts()}")
