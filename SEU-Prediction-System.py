# DOES NOT CURRENTLY WORK TO FULL CAPACITY

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time

class RealTimeSEUPredictor:
    """
    Real-time SEU prediction system for high-altitude balloon
    Runs on Raspberry Pi during flight
    """
    
    def __init__(self, trained_model=None):
        self.trained_model = trained_model  # Your pre-trained RandomForest
        self.data_buffer = []  # Rolling window of recent data
        self.buffer_size = 50  # Keep last 50 measurements
        self.prediction_horizon = 4  # Predict 4 samples ahead (1 minute)
        
        # Alert thresholds
        self.seu_alert_threshold = 10  # Alert if >10 SEUs predicted
        self.anomaly_threshold = -0.1  # Alert if anomaly score < -0.1
        
    def add_measurement(self, timestamp, altitude, temperature, bit_flips, 
                       max_run_length, gps_lat, gps_lon):
        """Add new measurement and make predictions"""
        
        # Create measurement record
        measurement = {
            'timestamp': timestamp,
            'altitude': altitude,
            'temperature': temperature,
            'bit_flips_count': bit_flips,
            'max_run_length': max_run_length,
            'latitude': gps_lat,
            'longitude': gps_lon
        }
        
        # Add to buffer
        self.data_buffer.append(measurement)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)  # Remove oldest
            
        # Make predictions if we have enough data
        if len(self.data_buffer) >= 10:
            predictions = self.predict_next_seus()
            alerts = self.check_alerts(predictions)
            
            return {
                'predictions': predictions,
                'alerts': alerts,
                'current_status': self.get_current_status()
            }
        
        return None
    
    def predict_next_seus(self):
        """Predict SEU events for next few time steps"""
        if len(self.data_buffer) < 10:
            return None
            
        # Convert buffer to DataFrame
        df = pd.DataFrame(self.data_buffer)
        
        # Calculate features
        df = self.calculate_features(df)
        
        predictions = {}
        
        # 1. Immediate prediction (next 15 seconds)
        current_features = self.extract_features(df.iloc[-1])
        immediate_pred = self.trained_model.predict([current_features])[0]
        predictions['immediate'] = {
            'time_ahead': '15 seconds',
            'predicted_seus': max(0, int(immediate_pred)),
            'confidence': self.calculate_confidence(current_features)
        }
        
        # 2. Short-term prediction (1 minute ahead)
        # Assume slight altitude increase and temperature drop
        future_features = current_features.copy()
        future_features[0] += 50  # altitude +50m
        future_features[1] -= 0.5  # temperature -0.5°C
        
        short_term_pred = self.trained_model.predict([future_features])[0]
        predictions['short_term'] = {
            'time_ahead': '1 minute',
            'predicted_seus': max(0, int(short_term_pred)),
            'confidence': self.calculate_confidence(future_features)
        }
        
        # 3. Anomaly detection on recent data
        recent_data = df.tail(5)
        anomaly_score = self.detect_anomaly(recent_data)
        predictions['anomaly_status'] = {
            'score': anomaly_score,
            'is_anomalous': anomaly_score < self.anomaly_threshold,
            'description': self.interpret_anomaly(anomaly_score)
        }
        
        return predictions
    
    def calculate_features(self, df):
        """Calculate derived features from raw measurements"""
        # Time deltas
        df['time_delta'] = df['timestamp'].diff().dt.total_seconds().fillna(15)
        
        # Rates
        df['bit_flip_rate'] = df['bit_flips_count'] / df['time_delta']
        df['altitude_change_rate'] = df['altitude'].diff() / df['time_delta']
        df['temperature_gradient'] = df['temperature'].diff()
        
        # Cosmic ray intensity
        df['cosmic_ray_intensity'] = 1.0 * np.exp(df['altitude'] / 4500)
        
        # Moving averages
        df['bit_flip_ma_5'] = df['bit_flips_count'].rolling(window=5, min_periods=1).mean()
        df['altitude_ma_10'] = df['altitude'].rolling(window=10, min_periods=1).mean()
        
        return df.fillna(0)
    
    def extract_features(self, row):
        """Extract feature vector for prediction"""
        return np.array([
            row['altitude'],
            row['temperature'],
            row['cosmic_ray_intensity'],
            row['altitude_change_rate'],
            row['bit_flip_ma_5']
        ])
    
    def detect_anomaly(self, recent_data):
        """Simple anomaly detection based on recent patterns"""
        if len(recent_data) < 3:
            return 0.0
            
        # Check if current SEU rate is much higher than recent average
        current_rate = recent_data['bit_flip_rate'].iloc[-1]
        avg_rate = recent_data['bit_flip_rate'].mean()
        
        if avg_rate > 0:
            ratio = current_rate / avg_rate
            # Convert to anomaly score (negative = more anomalous)
            return min(0.2, 0.1 - (ratio - 1) * 0.1)
        else:
            return 0.1
    
    def calculate_confidence(self, features):
        """Calculate prediction confidence based on feature stability"""
        # Higher confidence when conditions are stable
        altitude_rate = abs(features[3])  # altitude change rate
        
        if altitude_rate < 1.0:  # Stable conditions
            return 0.85
        elif altitude_rate < 5.0:  # Moderate change
            return 0.70
        else:  # Rapid change
            return 0.55
    
    def check_alerts(self, predictions):
        """Check if any alerts should be triggered"""
        alerts = []
        
        if predictions is None:
            return alerts
            
        # High SEU rate alerts
        immediate_seus = predictions['immediate']['predicted_seus']
        if immediate_seus > self.seu_alert_threshold:
            alerts.append({
                'type': 'HIGH_SEU_RATE',
                'severity': 'WARNING',
                'message': f'High SEU rate predicted: {immediate_seus} events in next 15s',
                'recommended_action': 'Monitor memory integrity, consider data backup'
            })
        
        # Anomaly alerts
        if predictions['anomaly_status']['is_anomalous']:
            alerts.append({
                'type': 'COSMIC_RAY_BURST',
                'severity': 'ALERT',
                'message': 'Cosmic ray shower detected - elevated SEU risk',
                'recommended_action': 'Increase data collection frequency, activate protection'
            })
        
        # Extreme altitude alerts
        current_altitude = self.data_buffer[-1]['altitude']
        if current_altitude > 30000:  # Above 30km
            alerts.append({
                'type': 'HIGH_ALTITUDE',
                'severity': 'INFO',
                'message': f'Extreme altitude reached: {current_altitude:.0f}m - peak SEU risk',
                'recommended_action': 'Maximum vigilance for memory errors'
            })
        
        return alerts
    
    def get_current_status(self):
        """Get current system status"""
        if not self.data_buffer:
            return None
            
        latest = self.data_buffer[-1]
        
        return {
            'altitude': latest['altitude'],
            'temperature': latest['temperature'],
            'recent_seus': latest['bit_flips_count'],
            'cosmic_intensity': 1.0 * np.exp(latest['altitude'] / 4500),
            'data_points_collected': len(self.data_buffer),
            'system_health': 'OPERATIONAL'
        }
    
    def interpret_anomaly(self, score):
        """Interpret anomaly score for human understanding"""
        if score > 0.05:
            return "Normal cosmic ray background"
        elif score > -0.05:
            return "Slightly elevated radiation"
        elif score > -0.15:
            return "Cosmic ray shower event detected"
        else:
            return "Major cosmic ray burst - extreme SEU risk"
    
    def save_prediction_log(self, predictions, filename='seu_predictions.json'):
        """Save predictions for analysis"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'buffer_size': len(self.data_buffer)
        }
        
        try:
            with open(filename, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Failed to save prediction log: {e}")

# Example usage during balloon flight
def simulate_realtime_prediction():
    """Simulate real-time prediction during balloon flight"""
    
    print("Real-Time SEU Prediction System")
    print("=" * 50)
    
    # Initialize predictor (you'd load your trained model here)
    predictor = RealTimeSEUPredictor()
    
    # Simulate receiving measurements every 15 seconds
    # (In real flight, this comes from your sensors)
    
    sample_measurements = [
        (datetime.now(), 15000, 5.0, 2, 1, 39.01, -98.00),
        (datetime.now() + timedelta(seconds=15), 15200, 4.8, 3, 2, 39.01, -98.00),
        (datetime.now() + timedelta(seconds=30), 15500, 4.5, 1, 1, 39.01, -98.00),
        (datetime.now() + timedelta(seconds=45), 16000, 4.0, 4, 2, 39.01, -98.00),
        (datetime.now() + timedelta(seconds=60), 16800, 3.2, 8, 3, 39.01, -98.00),  # SEU burst
        (datetime.now() + timedelta(seconds=75), 17200, 2.9, 12, 4, 39.01, -98.00), # Continuing
        (datetime.now() + timedelta(seconds=90), 17800, 2.5, 6, 2, 39.01, -98.00),  # Declining
    ]
    
    for i, (timestamp, alt, temp, flips, run_len, lat, lon) in enumerate(sample_measurements):
        print(f"\nMeasurement {i+1}: Alt={alt}m, Temp={temp}°C, SEUs={flips}")
        
        result = predictor.add_measurement(timestamp, alt, temp, flips, run_len, lat, lon)
        
        if result:
            # Display predictions
            pred = result['predictions']
            print(f"Immediate: {pred['immediate']['predicted_seus']} SEUs in 15s")
            print(f"Short-term: {pred['short_term']['predicted_seus']} SEUs in 1min")
            
            # Display alerts
            if result['alerts']:
                print("ALERTS:")
                for alert in result['alerts']:
                    print(f"   {alert['severity']}: {alert['message']}")
            else:
                print("No alerts - normal conditions")
        
        time.sleep(1)  # Simulate time between measurements

if __name__ == "__main__":
    simulate_realtime_prediction()
