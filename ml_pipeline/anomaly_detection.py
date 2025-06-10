import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

class AnomalyDetectionModel:
    """Advanced SEU anomaly detection model with training, evaluation, and persistence"""
    
    def __init__(self, model_path="models/seu_anomaly_model.pkl"):
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.anomaly_detector = None
        self.classifier = None
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Try to load existing model
        if os.path.exists(model_path):
            try:
                self.load_model()
                print(f"Model loaded from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                self.initialize_model()
        else:
            print("No existing model found. Will train new model.")
            self.initialize_model()
            
    def initialize_model(self):
        """Initialize model architecture"""
        # Anomaly detector for unlabeled data
        self.anomaly_detector = IsolationForest(
            contamination=0.05,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            random_state=42
        )
        
        # Classifier for labeled data (after we've collected some anomalies)
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    
    def save_model(self):
        """Save the trained model to disk"""
        model_package = {
            'anomaly_detector': self.anomaly_detector,
            'classifier': self.classifier,
            'scaler': self.scaler
        }
        joblib.dump(model_package, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model from disk"""
        model_package = joblib.load(self.model_path)
        self.anomaly_detector = model_package['anomaly_detector']
        self.classifier = model_package['classifier']
        self.scaler = model_package['scaler']
    
    def preprocess_data(self, df):
        """Extract and scale features for anomaly detection"""
        # Extract key features
        features = ['altitude', 'temperature', 'bit_flips_count', 'cosmic_intensity']
        
        # Add derived features if they exist
        if 'bit_flip_rate' in df.columns:
            features.append('bit_flip_rate')
        if 'max_run_length' in df.columns:
            features.append('max_run_length')
        if 'altitude_change_rate' in df.columns:
            features.append('altitude_change_rate')
        
        # Handle missing features by creating them
        if 'bit_flip_rate' not in df.columns and 'time_delta' in df.columns:
            df['bit_flip_rate'] = df['bit_flips_count'] / df['time_delta']
            features.append('bit_flip_rate')
        
        # Ensure all required columns exist
        for feature in features:
            if feature not in df.columns:
                print(f"Warning: Feature '{feature}' not found in data. Using placeholder values.")
                df[feature] = 0
        
        # Get features and scale
        X = df[features]
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, features
    
    def train_unsupervised(self, df):
        """Train the anomaly detection model on unlabeled data"""
        X_scaled, _ = self.preprocess_data(df)
          # Ensure the anomaly detector is initialized
        if self.anomaly_detector is None:
            self.initialize_model()
        if self.anomaly_detector is None:
            raise RuntimeError("Anomaly detector is not initialized.")
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        # Apply model to get anomaly scores
        df['anomaly_score'] = self.anomaly_detector.score_samples(X_scaled)
        df['is_anomaly'] = self.anomaly_detector.predict(X_scaled) == -1
        
        # Save trained model
        self.save_model()
        
        return df
    
    def train_supervised(self, df, label_column='is_anomaly'):
        """Train a supervised classifier using labeled data"""
        if label_column not in df.columns:
            print(f"Warning: Label column '{label_column}' not found in data.")
            return df
            
        X_scaled, _ = self.preprocess_data(df)
        y = df[label_column].astype(int)
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Ensure the classifier is initialized
        if self.classifier is None:
            self.initialize_model()
        if self.classifier is None:
            raise RuntimeError("Classifier is not initialized.")
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save trained model
        self.save_model()
        
        # Add predictions to dataframe
        df['anomaly_pred'] = self.classifier.predict(X_scaled)
        
        return df
    
    def optimize_model(self, df, label_column='is_anomaly'):
        """Optimize model hyperparameters using grid search"""
        if label_column not in df.columns:
            print(f"Warning: Label column '{label_column}' not found in data.")
            return
            
        X_scaled, _ = self.preprocess_data(df)
        y = df[label_column].astype(int)
        
        # Try different algorithms
        
        # Random Forest
        rf_pipe = Pipeline([
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        rf_params = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        # SVM
        svm_pipe = Pipeline([
            ('classifier', SVC(random_state=42, probability=True))
        ])
        
        svm_params = {
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': ['scale', 'auto', 0.1],
            'classifier__kernel': ['rbf', 'linear']
        }
        
        # Neural Network
        nn_pipe = Pipeline([
            ('classifier', MLPClassifier(random_state=42, max_iter=1000))
        ])
        
        nn_params = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate': ['constant', 'adaptive']
        }
        
        # Find best model
        best_score = 0
        best_model = None
        best_params = None
        
        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Evaluate models
        for name, pipe, params in [
            ('Random Forest', rf_pipe, rf_params),
            ('SVM', svm_pipe, svm_params),
            ('Neural Network', nn_pipe, nn_params)
        ]:
            print(f"\nOptimizing {name}...")
            grid = GridSearchCV(pipe, params, cv=5, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            print(f"Best score: {grid.best_score_:.4f}")
            print(f"Best parameters: {grid.best_params_}")            # Evaluate on test set
            y_pred = grid.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            # Access as dict with string keys
            test_f1 = 0
            if 'weighted avg' in report and 'f1-score' in report['weighted avg']:
                test_f1 = report['weighted avg']['f1-score']
            
            print(f"Test F1 Score: {test_f1:.4f}")
            
            if test_f1 > best_score:
                best_score = test_f1
                best_model = grid.best_estimator_
                best_params = grid.best_params_
        
        # Set best model as classifier
        print(f"\nBest model: {best_model}")
        print(f"Best F1 score: {best_score:.4f}")
        
        self.classifier = best_model
        
        # Save optimized model
        self.save_model()
    
    def detect_anomalies(self, df):
        """Detect anomalies in new data"""
        X_scaled, _ = self.preprocess_data(df)
        
        # Ensure the anomaly detector is initialized
        if self.anomaly_detector is None:
            self.initialize_model()
        if self.anomaly_detector is None:
            raise RuntimeError("Anomaly detector is not initialized.")
        
        # Get anomaly scores
        df['anomaly_score'] = self.anomaly_detector.score_samples(X_scaled)
        df['is_anomaly'] = self.anomaly_detector.predict(X_scaled) == -1
        
        # If classifier is trained, also get its predictions
        if self.classifier is not None and hasattr(self.classifier, 'predict_proba'):
            df['anomaly_probability'] = self.classifier.predict_proba(X_scaled)[:, 1]
        
        return df
    
    def predict_real_time(self, data_dict):
        """Make prediction on real-time data point"""
        # Convert single data point to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Preprocess
        X_scaled, _ = self.preprocess_data(df)
        
        # Get anomaly score
        if self.anomaly_detector is None:
            raise RuntimeError("Anomaly detector is not initialized.")
        anomaly_score = self.anomaly_detector.score_samples(X_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(X_scaled)[0] == -1
        
        # Get classifier prediction if available
        anomaly_prob = None
        if self.classifier is not None and hasattr(self.classifier, 'predict_proba'):
            anomaly_prob = self.classifier.predict_proba(X_scaled)[0, 1]
        
        return {
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly,
            'anomaly_probability': anomaly_prob
        }


# Example usage for training and evaluating the model
def train_and_evaluate():
    print("SEU Anomaly Detection Model Training")
    print("=" * 50)
    
    # Load data
    try:
        df = pd.read_csv("seu_synthetic_data.csv")
        print(f"Loaded {len(df)} data points")
    except FileNotFoundError:
        print("Error: Training data file not found!")
        return
    
    # Initialize model
    model = AnomalyDetectionModel()
    
    # Train unsupervised model
    print("\nTraining unsupervised anomaly detection model...")
    df = model.train_unsupervised(df)
    
    # Evaluate results
    anomaly_count = df['is_anomaly'].sum()
    print(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Scatter plot of altitude vs bit flips with anomalies highlighted
    plt.subplot(2, 2, 1)
    plt.scatter(df[~df['is_anomaly']]['altitude'], df[~df['is_anomaly']]['bit_flips_count'], 
              alpha=0.5, label='Normal')
    plt.scatter(df[df['is_anomaly']]['altitude'], df[df['is_anomaly']]['bit_flips_count'], 
              color='red', alpha=0.7, label='Anomaly')
    plt.xlabel('Altitude (m)')
    plt.ylabel('Bit Flips Count')
    plt.title('SEU Anomaly Detection')
    plt.legend()
    
    # Plot 2: Anomaly score histogram
    plt.subplot(2, 2, 2)
    plt.hist(df['anomaly_score'], bins=50, alpha=0.8)
    plt.axvline(float(np.percentile(df['anomaly_score'], 5)), color='red', linestyle='--', 
               label='Anomaly Threshold')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    
    # Plot 3: Bit flips over altitude with anomalies
    plt.subplot(2, 1, 2)
    plt.scatter(df.index, df['bit_flips_count'], c=df['is_anomaly'], cmap='coolwarm', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Bit Flips Count')
    plt.title('Time Series with Anomalies')
    cbar = plt.colorbar()
    cbar.set_label('Anomaly')
    
    plt.tight_layout()
    plt.savefig('seu_anomaly_detection.png', dpi=300)
    plt.show()
    
    # Train supervised model with detected anomalies
    print("\nTraining supervised classifier...")
    df = model.train_supervised(df)
    
    # Optimize model (commented out by default as it can be time-consuming)
    # print("\nOptimizing model (this may take a while)...")
    # model.optimize_model(df)
    
    # Print confusion matrix
    if 'anomaly_pred' in df.columns:
        print("\nConfusion Matrix:")
        print(confusion_matrix(df['is_anomaly'], df['anomaly_pred']))
    
    print("\nModel training complete!")
    return model


if __name__ == "__main__":
    model = train_and_evaluate()
