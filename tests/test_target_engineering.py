"""
Unit tests for target engineering module
"""

import pytest
import pandas as pd
import numpy as np
from src.target_engineering import RiskTargetEngineer, merge_target_with_data


def test_rfm_calculation():
    """Test RFM calculation"""
    # Create sample data
    df = pd.DataFrame({
        'CustomerId': ['A', 'A', 'B', 'B', 'C'],
        'TransactionStartTime': ['2024-01-01', '2024-01-15', '2024-01-10', '2024-01-20', '2024-01-05'],
        'Amount': [100, 200, 150, 250, 300]
    })
    
    engineer = RiskTargetEngineer(snapshot_date='2024-01-31')
    rfm = engineer.calculate_rfm(df)
    
    assert 'recency_days' in rfm.columns
    assert 'frequency' in rfm.columns
    assert 'monetary' in rfm.columns
    assert len(rfm) == 3  # 3 unique customers
    assert rfm[rfm['CustomerId'] == 'A']['frequency'].iloc[0] == 2


def test_clustering():
    """Test customer clustering"""
    # Create sample RFM data
    rfm_df = pd.DataFrame({
        'CustomerId': [f'C{i}' for i in range(100)],
        'recency_days': np.random.exponential(30, 100),
        'frequency': np.random.poisson(5, 100),
        'monetary': np.random.gamma(2, 500, 100)
    })
    
    engineer = RiskTargetEngineer()
    rfm_prepared, rfm_scaled = engineer.prepare_rfm_for_clustering(rfm_df)
    rfm_with_clusters, high_risk_cluster = engineer.cluster_customers(rfm_prepared, rfm_scaled, n_clusters=3)
    
    assert 'cluster' in rfm_with_clusters.columns
    assert len(rfm_with_clusters['cluster'].unique()) == 3
    assert 0 <= high_risk_cluster <= 2


def test_high_risk_assignment():
    """Test high-risk label assignment"""
    rfm_df = pd.DataFrame({
        'CustomerId': [f'C{i}' for i in range(100)],
        'recency_days': np.random.exponential(30, 100),
        'frequency': np.random.poisson(5, 100),
        'monetary': np.random.gamma(2, 500, 100)
    })
    
    engineer = RiskTargetEngineer()
    rfm_prepared, rfm_scaled = engineer.prepare_rfm_for_clustering(rfm_df)
    rfm_with_clusters, high_risk_cluster = engineer.cluster_customers(rfm_prepared, rfm_scaled)
    result = engineer.assign_high_risk_label(rfm_with_clusters, high_risk_cluster)
    
    assert 'is_high_risk' in result.columns
    assert result['is_high_risk'].sum() > 0
    assert result['is_high_risk'].sum() < len(result)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])