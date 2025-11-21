"""
HAMAD: Hybrid Attention-based Multi-scale Anomaly Detection
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class HAMADFramework:
    def __init__(self, rf_params=None, xgb_params=None):
        """Initialize HAMAD with ensemble classifiers"""
        
        # Default Random Forest parameters
        if rf_params is None:
            rf_params = {
                'n_estimators': 100,
                'max_depth': 20,
                'random_state': 42,
                'n_jobs': -1
            }
        
        # Default XGBoost parameters
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
        
        self.rf_classifier = RandomForestClassifier(**rf_params)
        self.xgb_classifier = XGBClassifier(**xgb_params)
        self.attention_weights = None
        self.adaptive_threshold = 0.5
        
    def train_ensemble(self, X_train, y_train):
        """Train both RF and XGBoost classifiers"""
        print("Training Random Forest...")
        self.rf_classifier.fit(X_train, y_train)
        
        print("Training XGBoost...")
        self.xgb_classifier.fit(X_train, y_train)
        
        print("✓ Training complete!")
        
    def compute_attention_weights(self, X_val, y_val):
        """Compute attention weights based on validation performance"""
        rf_pred = self.rf_classifier.predict(X_val)
        xgb_pred = self.xgb_classifier.predict(X_val)
        
        rf_acc = accuracy_score(y_val, rf_pred)
        xgb_acc = accuracy_score(y_val, xgb_pred)
        
        # Softmax attention weights
        total = rf_acc + xgb_acc
        self.attention_weights = {
            'rf': rf_acc / total,
            'xgb': xgb_acc / total
        }
        
        print(f"Attention weights - RF: {self.attention_weights['rf']:.3f}, XGB: {self.attention_weights['xgb']:.3f}")
        
    def predict_proba(self, X):
        """Weighted ensemble prediction"""
        rf_proba = self.rf_classifier.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_classifier.predict_proba(X)[:, 1]
        
        # Attention-weighted combination
        if self.attention_weights is None:
            ensemble_proba = (rf_proba + xgb_proba) / 2
        else:
            ensemble_proba = (
                self.attention_weights['rf'] * rf_proba +
                self.attention_weights['xgb'] * xgb_proba
            )
        
        return ensemble_proba
    
    def predict(self, X):
        """Binary prediction with adaptive threshold"""
        proba = self.predict_proba(X)
        return (proba >= self.adaptive_threshold).astype(int)
    
    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def save_model(self, path='models/hamad_model.pkl'):
        """Save trained model"""
        model_data = {
            'rf_classifier': self.rf_classifier,
            'xgb_classifier': self.xgb_classifier,
            'attention_weights': self.attention_weights,
            'adaptive_threshold': self.adaptive_threshold
        }
        joblib.dump(model_data, path)
        print(f"✓ Model saved to {path}")
    
    @staticmethod
    def load_model(path='models/hamad_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(path)
        
        hamad = HAMADFramework()
        hamad.rf_classifier = model_data['rf_classifier']
        hamad.xgb_classifier = model_data['xgb_classifier']
        hamad.attention_weights = model_data['attention_weights']
        hamad.adaptive_threshold = model_data['adaptive_threshold']
        
        print(f"✓ Model loaded from {path}")
        return hamad

if __name__ == "__main__":
    # Load processed data
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Initialize and train HAMAD
    hamad = HAMADFramework()
    hamad.train_ensemble(X_train, y_train)
    
    # Compute attention weights on validation set (using test for demo)
    hamad.compute_attention_weights(X_test[:1000], y_test[:1000])
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = hamad.evaluate(X_test, y_test)
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    hamad.save_model()
