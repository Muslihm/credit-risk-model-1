"""
Model Training and Tracking for Credit Risk/Fraud Detection
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Set environment variable to allow file store BEFORE importing mlflow
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import joblib

# Set MLflow tracking URI
mlflow.set_tracking_uri("mlruns")
os.makedirs("mlruns", exist_ok=True)


class CreditRiskModelTrainer:
    """Complete model training pipeline with MLflow integration"""
    
    def __init__(self, random_state=42, test_size=0.3):
        self.random_state = random_state
        self.test_size = test_size
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def clean_data(self, df):
        """Clean the data: remove non-numeric columns and handle strings"""
        df_clean = df.copy()
        
        # Drop CustomerId if it exists
        if 'CustomerId' in df_clean.columns:
            df_clean = df_clean.drop(columns=['CustomerId'])
        
        # Convert boolean columns to int (True/False -> 1/0)
        bool_cols = df_clean.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df_clean[col] = df_clean[col].astype(int)
        
        # For any remaining object/string columns, try to convert or drop
        string_cols = df_clean.select_dtypes(include=['object']).columns
        for col in string_cols:
            # Try to convert to numeric
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Fill any NaN with 0
        df_clean = df_clean.fillna(0)
        
        return df_clean
    
    def load_and_split_data(self, data_path=None):
        """Load processed data and split into train/test sets"""
        print("="*60)
        print("1. DATA PREPARATION")
        print("="*60)
        
        # Try multiple paths
        possible_paths = [
            'data/processed_data.csv',
            '../data/processed_data.csv',
            'processed_data.csv',
        ]
        
        if data_path:
            possible_paths.insert(0, data_path)
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                print(f"\n✅ Loading data from: {path}")
                df = pd.read_csv(path)
                break
        
        if df is None:
            raise FileNotFoundError("Could not find processed_data.csv")
        
        print(f"Original shape: {df.shape}")
        
        # Clean the data
        print("\nCleaning data (converting strings to numbers)...")
        df = self.clean_data(df)
        print(f"Cleaned shape: {df.shape}")
        
        # Separate features and target
        target_col = 'is_high_risk'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Drop any remaining non-numeric columns
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"Converting {col} to numeric...")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        print(f"\nTarget distribution:")
        print(f"  Class 0 (Low Risk): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
        print(f"  Class 1 (High Risk): {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nTrain set: {self.X_train.shape[0]:,} samples")
        print(f"Test set: {self.X_test.shape[0]:,} samples")
        print(f"Features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model and return metrics"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics, y_pred, y_pred_proba
    
    def _log_confusion_matrix(self, y_test, y_pred, model_name):
        """Create and log confusion matrix plot"""
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Low Risk', 'High Risk'],
                    yticklabels=['Low Risk', 'High Risk'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        os.makedirs('plots', exist_ok=True)
        plot_path = f'plots/confusion_matrix_{model_name}.png'
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def _log_roc_curve(self, y_test, y_pred_proba, model_name):
        """Create and log ROC curve plot"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        
        plot_path = f'plots/roc_curve_{model_name}.png'
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*60)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*60)
        
        with mlflow.start_run(run_name="Logistic_Regression", nested=True):
            model = LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000,
                C=0.1
            )
            model.fit(self.X_train, self.y_train)
            
            metrics, y_pred, y_pred_proba = self._evaluate_model(model, self.X_test, self.y_test, "Logistic Regression")
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(self._log_confusion_matrix(self.y_test, y_pred, "Logistic_Regression"))
            mlflow.log_artifact(self._log_roc_curve(self.y_test, y_pred_proba, "Logistic_Regression"))
            
            signature = infer_signature(self.X_train, model.predict(self.X_train))
            mlflow.sklearn.log_model(model, "logistic_regression_model", signature=signature)
            
            print("\nMetrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            self.models['Logistic Regression'] = model
            self.results['Logistic Regression'] = metrics
            
            return model, metrics
    
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)
        
        with mlflow.start_run(run_name="Random_Forest", nested=True):
            model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_estimators=100,
                max_depth=10,
                n_jobs=-1
            )
            model.fit(self.X_train, self.y_train)
            
            metrics, y_pred, y_pred_proba = self._evaluate_model(model, self.X_test, self.y_test, "Random Forest")
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(self._log_confusion_matrix(self.y_test, y_pred, "Random_Forest"))
            mlflow.log_artifact(self._log_roc_curve(self.y_test, y_pred_proba, "Random_Forest"))
            
            signature = infer_signature(self.X_train, model.predict(self.X_train))
            mlflow.sklearn.log_model(model, "random_forest_model", signature=signature)
            
            print("\nMetrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            self.models['Random Forest'] = model
            self.results['Random Forest'] = metrics
            
            return model, metrics
    
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST")
        print("="*60)
        
        with mlflow.start_run(run_name="XGBoost", nested=True):
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            
            model = XGBClassifier(
                random_state=self.random_state,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                use_label_encoder=False,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                verbosity=0
            )
            model.fit(self.X_train, self.y_train)
            
            metrics, y_pred, y_pred_proba = self._evaluate_model(model, self.X_test, self.y_test, "XGBoost")
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(self._log_confusion_matrix(self.y_test, y_pred, "XGBoost"))
            mlflow.log_artifact(self._log_roc_curve(self.y_test, y_pred_proba, "XGBoost"))
            
            signature = infer_signature(self.X_train, model.predict(self.X_train))
            mlflow.sklearn.log_model(model, "xgboost_model", signature=signature)
            
            print("\nMetrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            self.models['XGBoost'] = model
            self.results['XGBoost'] = metrics
            
            return model, metrics
    
    def train_all_models(self):
        """Train all models and compare results"""
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        mlflow.set_experiment("Credit_Risk_Model_Experiment")
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        self.select_best_model()
        
        return self.results
    
    def select_best_model(self):
        """Select best model based on ROC-AUC score"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison = pd.DataFrame(self.results).T
        comparison = comparison.sort_values('roc_auc', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(comparison.round(4))
        
        self.best_model_name = comparison.index[0]
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   ROC-AUC: {comparison.loc[self.best_model_name, 'roc_auc']:.4f}")
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.best_model, 'models/best_model.pkl')
        print(f"\nBest model saved to: models/best_model.pkl")
        
        return self.best_model, self.best_model_name
    
    def generate_report(self):
        """Generate model evaluation report"""
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        report = f"""
CREDIT RISK MODEL EVALUATION REPORT
===================================

BEST MODEL: {self.best_model_name}

Performance Metrics:
"""
        for metric, value in self.results[self.best_model_name].items():
            report += f"  {metric}: {value:.4f}\n"
        
        os.makedirs('reports', exist_ok=True)
        with open('reports/model_evaluation_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        return report


def main():
    print("="*60)
    print("CREDIT RISK MODEL TRAINING - TASK 5")
    print("="*60)
    
    trainer = CreditRiskModelTrainer(random_state=42, test_size=0.3)
    trainer.load_and_split_data()
    trainer.train_all_models()
    trainer.generate_report()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✅ Best model: {trainer.best_model_name}")
    print(f"✅ Model saved to: models/best_model.pkl")
    
    return trainer


if __name__ == "__main__":
    trainer = main()
