"""
Data preprocessing and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self, dataset_name='nsl-kdd'):
        self.dataset_name = dataset_name
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        
    def load_nsl_kdd(self):
        """Load NSL-KDD dataset"""
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
            'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]
        
        train_path = 'data/raw/nsl-kdd/KDDTrain+.txt'
        test_path = 'data/raw/nsl-kdd/KDDTest+.txt'
        
        df_train = pd.read_csv(train_path, names=columns)
        df_test = pd.read_csv(test_path, names=columns)
        
        df = pd.concat([df_train, df_test], ignore_index=True)
        
        return df
    
    def extract_features(self, df):
        """Feature engineering"""
        # Separate features and labels
        X = df.drop(['label', 'difficulty'], axis=1)
        y = df['label']
        
        # Binary classification: normal vs attack
        y_binary = y.apply(lambda x: 0 if x == 'normal' else 1)
        
        # Encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        X_encoded = pd.get_dummies(X, columns=categorical_cols)
        
        return X_encoded, y_binary, y
    
    def normalize_features(self, X_train, X_test):
        """Normalize features using Min-Max scaling"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self):
        """Complete preprocessing pipeline"""
        print(f"Loading {self.dataset_name} dataset...")
        
        if self.dataset_name == 'nsl-kdd':
            df = self.load_nsl_kdd()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported yet")
        
        print(f"Dataset shape: {df.shape}")
        
        # Feature extraction
        print("Extracting features...")
        X, y_binary, y_multi = self.extract_features(df)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Normalization
        print("Normalizing features...")
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        
        # Convert back to DataFrame
        X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        # Save processed data
        os.makedirs('data/processed', exist_ok=True)
        X_train_df.to_csv('data/processed/X_train.csv', index=False)
        X_test_df.to_csv('data/processed/X_test.csv', index=False)
        pd.DataFrame(y_train).to_csv('data/processed/y_train.csv', index=False)
        pd.DataFrame(y_test).to_csv('data/processed/y_test.csv', index=False)
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        print(f"✓ Preprocessing complete!")
        print(f"  Train samples: {len(X_train_df)}")
        print(f"  Test samples: {len(X_test_df)}")
        print(f"  Features: {X_train_df.shape[1]}")
        
        return X_train_df, X_test_df, y_train, y_test

if __name__ == "__main__":
    preprocessor = DataPreprocessor(dataset_name='nsl-kdd')
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
