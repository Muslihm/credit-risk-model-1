"""
Simplified Credit Risk Data Processing Pipeline
Transforms raw data into model-ready format
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


# ============================================
# 1. TIME-BASED FEATURES
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


# ============================================
# 2. HANDLE MISSING VALUES
# ============================================

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


# ============================================
# 3. ENCODE CATEGORICAL VARIABLES
# ============================================

class EncodeCategorical(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        # Remove ID columns
        id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
        self.cat_cols = [c for c in self.cat_cols if c not in id_cols]
        # Only encode columns with <= 20 unique values
        self.cat_cols = [c for c in self.cat_cols if X[c].nunique() <= 20]
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        if len(self.cat_cols) > 0:
            encoded = pd.get_dummies(df[self.cat_cols], drop_first=True)
            df = df.drop(columns=self.cat_cols)
            df = pd.concat([df, encoded], axis=1)
        return df


# ============================================
# 4. SCALE NUMERICAL FEATURES
# ============================================

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


# ============================================
# 5. DROP ID COLUMNS
# ============================================

class DropIdColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.id_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        df = X.copy()
        cols_to_drop = [c for c in self.id_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        return df


# ============================================
# 6. CREATE COMPLETE PIPELINE
# ============================================

def create_pipeline():
    """Create complete data processing pipeline"""
    return Pipeline([
        ('drop_ids', DropIdColumns()),
        ('time_features', ExtractTimeFeatures()),
        ('handle_missing', HandleMissingValues()),
        ('encode_categorical', EncodeCategorical()),
        ('scale_features', ScaleFeatures())
    ])


# ============================================
# 7. MAIN EXECUTION
# ============================================

def main():
    print("= - data_processing.py:151"*60)
    print("CREDIT RISK DATA PROCESSING PIPELINE - data_processing.py:152")
    print("= - data_processing.py:153"*60)
    
    # Load data
    print("\n1. Loading data... - data_processing.py:156")
    df = pd.read_excel('../data/data.xlsx')
    print(f"✓ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns - data_processing.py:158")
    
    # Split features and target
    target_col = 'FraudResult'
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        print(f"\n   Target: {target_col} - data_processing.py:165")
        print(f"Fraud rate: {y.mean():.2%} - data_processing.py:166")
    else:
        X = df
        y = None
        print("\n   No target column found - data_processing.py:170")
    
    # Create and apply pipeline
    print("\n2. Creating pipeline... - data_processing.py:173")
    pipeline = create_pipeline()
    
    print("\n3. Processing data... - data_processing.py:176")
    X_processed = pipeline.fit_transform(X, y)
    
    print(f"\n4. Results: - data_processing.py:179")
    print(f"Original shape: {X.shape} - data_processing.py:180")
    print(f"Processed shape: {X_processed.shape} - data_processing.py:181")
    print(f"Features created: {X_processed.shape[1]} - data_processing.py:182")
    
    print("\n5. Sample of processed data (first 5 rows, first 5 columns): - data_processing.py:184")
    print(X_processed.iloc[:5, :5])
    
    print("\n - data_processing.py:187" + "="*60)
    print("✓ SUCCESS! Pipeline ready for model training - data_processing.py:188")
    print("= - data_processing.py:189"*60)
    
    return X_processed, y, pipeline


if __name__ == "__main__":
    X, y, pipeline = main()