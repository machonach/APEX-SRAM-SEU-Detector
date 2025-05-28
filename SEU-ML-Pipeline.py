import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class SEUDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.seu_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        
    def preprocess_data(self, df):
        """Preprocess the data and create derived features"""
        # Create derived features
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
    
    def detect_anomalies(self, df):
        """Detect SEU anomalies using isolation forest"""
        features = ['bit_flips_count', 'max_run_length', 'altitude', 'temperature', 
                   'bit_flip_rate', 'cosmic_ray_intensity']
        
        X = df[features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Detect anomalies
        anomaly_labels = self.anomaly_detector.fit_predict(X_scaled)
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        
        df['anomaly_label'] = anomaly_labels  # -1 for anomaly, 1 for normal
        df['anomaly_score'] = anomaly_scores  # Lower scores = more anomalous
        
        return df
    
    def predict_seu_frequency(self, df):
        """Predict future SEU frequency based on environmental conditions"""
        features = ['altitude', 'temperature', 'cosmic_ray_intensity', 
                   'altitude_change_rate', 'bit_flip_ma_5']
        
        X = df[features].fillna(0)
        y = df['bit_flips_count']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.seu_predictor.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.seu_predictor.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        df['predicted_seu_count'] = self.seu_predictor.predict(X)
        
        return df, {'mse': mse, 'r2': r2}
    
    def cluster_seu_events(self, df):
        """Cluster SEU events to identify patterns"""
        anomaly_data = df[df['anomaly_label'] == -1]
        
        if len(anomaly_data) > 0:
            features = ['altitude', 'bit_flips_count', 'max_run_length', 'temperature']
            X_anomaly = anomaly_data[features]
            
            clusters = self.clusterer.fit_predict(X_anomaly)
            anomaly_data['cluster'] = clusters
            
            return anomaly_data
        else:
            return pd.DataFrame()
    
    def analyze_seu_patterns(self, df):
        """Comprehensive analysis of SEU patterns"""
        results = {}
        
        # Basic statistics
        results['total_seus'] = df['bit_flips_count'].sum()
        results['peak_altitude'] = df['altitude'].max()
        results['max_seu_rate'] = df['bit_flip_rate'].max()
        results['anomaly_count'] = len(df[df['anomaly_label'] == -1])
        
        # Altitude correlation
        altitude_corr = df['bit_flips_count'].corr(df['altitude'])
        results['altitude_correlation'] = altitude_corr
        
        # Critical altitude analysis (where SEUs start increasing significantly)
        high_seu_threshold = df['bit_flips_count'].quantile(0.8)
        critical_altitude = df[df['bit_flips_count'] >= high_seu_threshold]['altitude'].min()
        results['critical_altitude'] = critical_altitude
        
        return results
    
    def visualize_results(self, df, save_plots=True):
        """Create comprehensive visualizations"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. SEU Count vs Altitude
        axes[0, 0].scatter(df['altitude'], df['bit_flips_count'], 
                          c=df['anomaly_score'], cmap='coolwarm', alpha=0.7)
        axes[0, 0].set_xlabel('Altitude (m)')
        axes[0, 0].set_ylabel('Bit Flips Count')
        axes[0, 0].set_title('SEU Events vs Altitude')
        
        # 2. Time series of SEU events
        axes[0, 1].plot(df.index, df['bit_flips_count'], label='Actual', alpha=0.7)
        axes[0, 1].plot(df.index, df['predicted_seu_count'], label='Predicted', alpha=0.7)
        axes[0, 1].fill_between(df.index, 0, df['bit_flips_count'], 
                               where=(df['anomaly_label'] == -1), 
                               color='red', alpha=0.3, label='Anomalies')
        axes[0, 1].set_xlabel('Time (samples)')
        axes[0, 1].set_ylabel('SEU Count')
        axes[0, 1].set_title('SEU Time Series with Predictions')
        axes[0, 1].legend()
        
        # 3. Altitude vs Temperature vs SEU Rate
        scatter = axes[0, 2].scatter(df['altitude'], df['temperature'], 
                                   c=df['bit_flip_rate'], cmap='plasma', s=50)
        axes[0, 2].set_xlabel('Altitude (m)')
        axes[0, 2].set_ylabel('Temperature (°C)')
        axes[0, 2].set_title('Environmental Conditions vs SEU Rate')
        plt.colorbar(scatter, ax=axes[0, 2], label='SEU Rate')
        
        # 4. Distribution of bit flip counts
        axes[1, 0].hist(df['bit_flips_count'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(df['bit_flips_count'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {df["bit_flips_count"].mean():.2f}')
        axes[1, 0].set_xlabel('Bit Flips Count')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of SEU Events')
        axes[1, 0].legend()
        
        # 5. Anomaly scores
        normal_data = df[df['anomaly_label'] == 1]
        anomaly_data = df[df['anomaly_label'] == -1]
        
        axes[1, 1].hist(normal_data['anomaly_score'], bins=30, alpha=0.7, 
                       label='Normal', color='blue')
        axes[1, 1].hist(anomaly_data['anomaly_score'], bins=30, alpha=0.7, 
                       label='Anomaly', color='red')
        axes[1, 1].set_xlabel('Anomaly Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Anomaly Score Distribution')
        axes[1, 1].legend()
        
        # 6. Max run length analysis
        axes[1, 2].scatter(df['altitude'], df['max_run_length'], 
                          c=df['bit_flips_count'], cmap='viridis', alpha=0.7)
        axes[1, 2].set_xlabel('Altitude (m)')
        axes[1, 2].set_ylabel('Max Run Length')
        axes[1, 2].set_title('Bit Flip Run Length vs Altitude')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('seu_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

# Main execution function
def run_seu_analysis(data_file='seu_synthetic_data.csv'):
    """Run the complete SEU analysis pipeline"""
    
    print("SRAM SEU Detector - ML Pipeline")
    print("=" * 50)
    
    # Load data
    print("Loading synthetic SEU data...")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} data points")
    
    # Initialize detector
    detector = SEUDetector()
    
    # Preprocess data
    print("Preprocessing data and creating features...")
    df = detector.preprocess_data(df)
    
    # Detect anomalies
    print("Detecting SEU anomalies...")
    df = detector.detect_anomalies(df)
    
    # Predict SEU frequency
    print("Training SEU prediction model...")
    df, prediction_metrics = detector.predict_seu_frequency(df)
    
    # Cluster SEU events
    print("Clustering SEU events...")
    anomaly_clusters = detector.cluster_seu_events(df)
    
    # Analyze patterns
    print("Analyzing SEU patterns...")
    analysis_results = detector.analyze_seu_patterns(df)
    
    # Print results
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Total SEU Events: {analysis_results['total_seus']}")
    print(f"Peak Altitude: {analysis_results['peak_altitude']:.0f} m")
    print(f"Maximum SEU Rate: {analysis_results['max_seu_rate']:.4f} flips/second")
    print(f"Anomalous Events Detected: {analysis_results['anomaly_count']}")
    print(f"Altitude-SEU Correlation: {analysis_results['altitude_correlation']:.3f}")
    print(f"Critical Altitude (SEU increase): {analysis_results['critical_altitude']:.0f} m")
    print(f"Prediction Model R²: {prediction_metrics['r2']:.3f}")
    print(f"Prediction RMSE: {np.sqrt(prediction_metrics['mse']):.3f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    detector.visualize_results(df)
    
    # Save processed data
    df.to_csv('seu_analysis_results.csv', index=False)
    print("\nResults saved to 'seu_analysis_results.csv'")
    
    return df, detector, analysis_results

if __name__ == "__main__":
    # Generate synthetic data first (if not exists)
    try:
        df = pd.read_csv('seu_synthetic_data.csv')
        print("Found existing synthetic data file")
    except FileNotFoundError:
        print("Synthetic data file not found. Please run the data generator first.")
        exit(1)
    
    # Run analysis
    results_df, seu_detector, results = run_seu_analysis()
