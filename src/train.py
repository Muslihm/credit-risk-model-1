"""
Model training pipeline for credit risk scoring.
Supports Logistic Regression (interpretable) and XGBoost (high performance).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib
from pathlib import Path
import json
from typing import Dict, Any, Tuple

from src.data_processing import prepare_training_data, WeightOfEvidenceEncoder, select_features_by_iv


class CreditRiskModel:
    """Wrapper class for credit risk models with unified interface."""
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Args:
            model_type: Either 'logistic_regression' or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.woe_encoder = None
        self.scaler = None
        self.selected_features = None
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
              use_woe: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Train the credit risk model.
        
        Args:
            X: Feature dataframe
            y: Target series
            use_woe: Apply WoE transformation (only for logistic regression)
            **kwargs: Additional model parameters
        
        Returns:
            Dictionary with training metrics
        """
        # Feature selection using IV
        feature_cols = X.columns.tolist()
        self.selected_features = select_features_by_iv(
            pd.concat([X, y], axis=1), 
            y.name if hasattr(y, 'name') else 'target',
            feature_cols,
            iv_threshold=kwargs.get('iv_threshold', 0.02)
        )
        
        X_filtered = X[self.selected_features].copy()
        
        # Apply WoE for logistic regression
        if self.model_type == 'logistic_regression' and use_woe:
            print("Applying Weight of Evidence transformation... - train.py:62")
            self.woe_encoder = WeightOfEvidenceEncoder()
            X_transformed = self.woe_encoder.fit_transform(X_filtered, y)
            
            # Scale features
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_transformed)
            
            # Train logistic regression
            C = kwargs.get('C', 1.0)
            self.model = LogisticRegression(C=C, max_iter=1000, random_state=42)
            
        elif self.model_type == 'xgboost':
            print("Training XGBoost model... - train.py:75")
            params = {
                'n_estimators': kwargs.get('n_estimators', 100),
                'max_depth': kwargs.get('max_depth', 6),
                'learning_rate': kwargs.get('learning_rate', 0.1),
                'subsample': kwargs.get('subsample', 0.8),
                'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
                'random_state': 42,
                'eval_metric': 'auc'
            }
            self.model = xgb.XGBClassifier(**params)
            X_processed = X_filtered  # XGBoost handles raw features
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train the model
        self.model.fit(X_processed, y)
        
        # Calculate metrics
        y_pred_proba = self.predict_proba(X)
        metrics = self._calculate_metrics(y, y_pred_proba)
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict default probability."""
        X_filtered = X[self.selected_features].copy()
        
        if self.model_type == 'logistic_regression' and self.woe_encoder is not None:
            X_transformed = self.woe_encoder.transform(X_filtered)
            X_processed = self.scaler.transform(X_transformed)
        else:
            X_processed = X_filtered
        
        return self.model.predict_proba(X_processed)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary default outcome."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate model performance metrics."""
        auc = roc_auc_score(y_true, y_pred_proba)
        # Use 0.5 threshold for confusion matrix
        y_pred = (y_pred_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'model_type': self.model_type,
            'auc_roc': auc,
            'gini': 2 * auc - 1,  # Gini = 2*AUC - 1
            'confusion_matrix': cm.tolist(),
            'accuracy': (cm[0,0] + cm[1,1]) / cm.sum()
        }
    
    def save(self, path: str):
    """Save model and preprocessing objects."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        artifacts = {
            'model': self.model,
            'model_type': self.model_type,
            'selected_features': self.selected_features,
            'woe_encoder': self.woe_encoder,
            'scaler': self.scaler
        }
        joblib.dump(artifacts, save_path)
        print(f"Model saved to {save_path} - train.py:146")
    
    def load(self, path: str):
        """Load model and preprocessing objects."""
        artifacts = joblib.load(path)
        self.model = artifacts['model']
        self.model_type = artifacts['model_type']
        self.selected_features = artifacts['selected_features']
        self.woe_encoder = artifacts['woe_encoder']
        self.scaler = artifacts['scaler']
        print(f"Model loaded from {path} - train.py:156")


def main():
    """Example training pipeline."""
    print("= - train.py:161" * 50)
    print("Credit Risk Model Training Pipeline - train.py:162")
    print("= - train.py:163" * 50)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 10000
    
    data = pd.DataFrame({
        'income': np.random.normal(50000, 20000, n_samples),
        'debt': np.random.normal(20000, 15000, n_samples),
        'credit_age_years': np.random.exponential(10, n_samples),
        'num_delinquencies': np.random.poisson(0.5, n_samples),
        'credit_utilization': np.random.beta(2, 5, n_samples),
        'num_credit_cards': np.random.poisson(2, n_samples),
        'default': np.random.binomial(1, 0.1, n_samples)  # 10% default rate
    })
    
    # Prepare data
    X, y = prepare_training_data(data, target_col='default')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape} - train.py:187")
    print(f"Test set: {X_test.shape} - train.py:188")
    print(f"Default rate  Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%} - train.py:189")
    
    # Train interpretable model (Logistic Regression with WoE)
    print("\n - train.py:192" + "=" * 30)
    print("Training Logistic Regression (Interpretable) - train.py:193")
    print("= - train.py:194" * 30)
    
    lr_model = CreditRiskModel(model_type='logistic_regression')
    lr_metrics = lr_model.train(X_train, y_train, use_woe=True, C=0.1)
    
    print(f"\nLogistic Regression Results: - train.py:199")
    print(f"AUCROC: {lr_metrics['auc_roc']:.4f} - train.py:200")
    print(f"Gini: {lr_metrics['gini']:.4f} - train.py:201")
    print(f"Accuracy: {lr_metrics['accuracy']:.4f} - train.py:202")
    
    # Test performance
    y_test_proba = lr_model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_test_proba)
    print(f"Test AUCROC: {test_auc:.4f} - train.py:207")
    
    # Train high-performance model (XGBoost)
    print("\n - train.py:210" + "=" * 30)
    print("Training XGBoost (High Performance) - train.py:211")
    print("= - train.py:212" * 30)
    
    xgb_model = CreditRiskModel(model_type='xgboost')
    xgb_metrics = xgb_model.train(X_train, y_train, n_estimators=100, max_depth=5)
    
    print(f"\nXGBoost Results: - train.py:217")
    print(f"AUCROC: {xgb_metrics['auc_roc']:.4f} - train.py:218")
    print(f"Gini: {xgb_metrics['gini']:.4f} - train.py:219")
    print(f"Accuracy: {xgb_metrics['accuracy']:.4f} - train.py:220")
    
    # Test performance
    y_test_proba_xgb = xgb_model.predict_proba(X_test)
    test_auc_xgb = roc_auc_score(y_test, y_test_proba_xgb)
    print(f"Test AUCROC: {test_auc_xgb:.4f} - train.py:225")
    # Save models
    lr_model.save('models/logistic_regression_model.joblib')
    xgb_model.save('models/xgboost_model.joblib')
    print("\n - train.py:229" + "=" * 50)
    print("Training complete! Models saved to 'models/' directory - train.py:230")
    print("= - train.py:231" * 50)
    return lr_model, xgb_model 

if __name__ == "__main__":
    
    main()

