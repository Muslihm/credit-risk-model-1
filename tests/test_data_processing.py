"""
Unit tests for data processing module
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to path so src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from src
from src.data_processing import (
    ExtractTimeFeatures,
    HandleMissingValues,
    EncodeCategorical,
    ScaleFeatures,
    DropIdColumns,
    RiskTargetEngineer
)


class TestFeatureEngineering:
    """Test feature engineering components"""
    
    def test_extract_time_features(self):
        """Test that time features are correctly extracted"""
        df = pd.DataFrame({
            'TransactionStartTime': ['2024-01-15 14:30:00', '2024-03-20 09:15:00']
        })
        
        transformer = ExtractTimeFeatures()
        result = transformer.fit_transform(df)
        
        # Check expected columns
        assert 'hour' in result.columns
        assert 'day_of_month' in result.columns
        assert 'month' in result.columns
        assert 'year' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'is_business_hour' in result.columns
        
        # Check values
        assert result['hour'].iloc[0] == 14
        assert result['is_business_hour'].iloc[0] == 1
    
    def test_handle_missing_values(self):
        """Test missing value imputation"""
        df = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'cat_col': ['A', 'B', np.nan, 'A', 'B']
        })
        
        transformer = HandleMissingValues()
        result = transformer.fit_transform(df)
        
        # No missing values should remain
        assert result.isnull().sum().sum() == 0
        
        # Numeric column should have median (3.0)
        assert result['numeric_col'].iloc[2] == 3.0
    
    def test_encode_categorical(self):
        """Test one-hot encoding of categorical variables"""
        df = pd.DataFrame({
            'color': ['red', 'blue', 'red', 'green'],
            'size': ['S', 'M', 'L', 'M']
        })
        
        transformer = EncodeCategorical()
        result = transformer.fit_transform(df)
        
        # Original columns should be dropped
        assert 'color' not in result.columns
        assert 'size' not in result.columns
        
        # Encoded columns should exist
        encoded_cols = [c for c in result.columns if c.startswith('color_') or c.startswith('size_')]
        assert len(encoded_cols) > 0
    
    def test_scale_features(self):
        """Test feature standardization"""
        df = pd.DataFrame({
            'amount': [100, 200, 300, 400, 500],
            'value': [10, 20, 30, 40, 50]
        })
        
        transformer = ScaleFeatures()
        result = transformer.fit_transform(df)
        
        # After standardization, mean should be ~0, std ~1
        assert abs(result['amount'].mean()) < 1e-10
        assert abs(result['amount'].std() - 1) < 0.1
    
    def test_drop_id_columns(self):
        """Test dropping of ID columns"""
        df = pd.DataFrame({
            'TransactionId': [1, 2, 3],
            'CustomerId': [100, 101, 102],
            'Amount': [10, 20, 30],
            'Value': [100, 200, 300]
        })
        
        transformer = DropIdColumns()
        result = transformer.fit_transform(df)
        
        # ID columns should be dropped
        assert 'TransactionId' not in result.columns
        assert 'CustomerId' not in result.columns
        
        # Data columns should remain
        assert 'Amount' in result.columns
        assert 'Value' in result.columns


class TestTargetEngineering:
    """Test target variable engineering"""
    
    def test_rfm_calculation(self):
        """Test RFM metrics calculation"""
        df = pd.DataFrame({
            'CustomerId': ['A', 'A', 'B', 'B', 'C'],
            'TransactionStartTime': ['2024-01-01', '2024-01-15', '2024-01-10', '2024-01-20', '2024-01-05'],
            'Amount': [100, 200, 150, 250, 300]
        })
        
        engineer = RiskTargetEngineer(random_state=42)
        rfm = engineer.calculate_rfm(df)
        
        assert 'recency_days' in rfm.columns
        assert 'frequency' in rfm.columns
        assert 'monetary' in rfm.columns
        assert len(rfm) == 3
    
    def test_clustering_output(self):
        """Test that clustering produces valid outputs"""
        np.random.seed(42)
        rfm_df = pd.DataFrame({
            'CustomerId': [f'C{i}' for i in range(50)],
            'recency_days': np.random.exponential(30, 50),
            'frequency': np.random.poisson(5, 50),
            'monetary': np.random.gamma(2, 500, 50)
        })
        
        engineer = RiskTargetEngineer(random_state=42)
        rfm_prepared, rfm_scaled = engineer.prepare_for_clustering(rfm_df)
        result = engineer.cluster_and_label(rfm_prepared, rfm_scaled)
        
        assert 'CustomerId' in result.columns
        assert 'is_high_risk' in result.columns
        assert result['is_high_risk'].isin([0, 1]).all()


class TestProcessedData:
    """Test the final processed data"""
    
    def test_processed_data_has_target_column(self):
        """Test that processed data contains the is_high_risk target column"""
        paths = ['data/processed_data.csv', '../data/processed_data.csv']
        df = None
        for path in paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        
        if df is None:
            pytest.skip("Processed data not found - run src/data_processing.py first")
        
        assert 'is_high_risk' in df.columns
        assert df['is_high_risk'].isin([0, 1]).all()
    
    def test_no_missing_values_in_processed_data(self):
        """Test that processed data has no missing values"""
        paths = ['data/processed_data.csv', '../data/processed_data.csv']
        df = None
        for path in paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        
        if df is None:
            pytest.skip("Processed data not found - run src/data_processing.py first")
        
        assert df.isnull().sum().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
