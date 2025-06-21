import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SEUDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.seu_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feature_columns = ['bit_flips_count', 'max_run_length', 'altitude', 
                              'temperature', 'bit_flip_rate', 'cosmic_ray_intensity']
    
    def preprocess_data(self, df):
        """Preprocess the data and create derived features"""
        # Create derived features
        if 'time_delta' not in df.columns:
            df['time_delta'] = 600  # Default 10-minute intervals
            
        df['bit_flip_rate'] = df['bit_flips_count'] / df['time_delta']
        df['altitude_change_rate'] = df['altitude'].diff() / df['time_delta']
        df['temperature_gradient'] = df['temperature'].diff()
        df['cosmic_ray_intensity'] = self._calculate_cosmic_intensity(df['altitude'])
        
        # Rolling averages for trend detection
        df['bit_flip_ma_5'] = df['bit_flips_count'].rolling(window=5, min_periods=1).mean()
        df['altitude_ma_10'] = df['altitude'].rolling(window=10, min_periods=1).mean()
        
        # Fill NaN values using forward fill then zeros
        df = df.ffill().fillna(0)
        
        return df
    
    def _calculate_cosmic_intensity(self, altitude):
        """Calculate expected cosmic ray intensity based on altitude"""
        # Cosmic ray intensity approximately doubles every 1.5km above 3km
        sea_level_intensity = 1.0
        return sea_level_intensity * np.exp((altitude - 0) / 4500)
    
    def train(self, train_df):
        """Train the ML models on the provided data"""
        X = train_df[self.feature_columns]
        y = train_df['bit_flips_count']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        # Train SEU predictor
        self.seu_predictor.fit(X_scaled, y)
    
    def predict(self, test_df):
        """Make predictions on test data"""
        X = test_df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Get SEU count predictions
        predictions = self.seu_predictor.predict(X_scaled)
        
        return predictions
    
    def detect_anomalies(self, df):
        """Detect anomalies in the data"""
        X = df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # -1 for anomalies, 1 for normal points
        anomaly_labels = self.anomaly_detector.predict(X_scaled)
        
        return anomaly_labels == -1  # Return boolean array where True indicates anomalies
