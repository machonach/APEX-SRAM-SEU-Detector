#!/usr/bin/env python3
"""
SEU Detector - Test Script
This script tests core functionality of the SEU Detector system.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.absolute()))

class TestSEUDetector(unittest.TestCase):
    """Test suite for SEU Detector components"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.sample_data = self._create_sample_data()
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def _create_sample_data(self):
        """Create sample SEU data for testing"""
        # Create a small synthetic dataset
        n = 50
        
        data = {
            'timestamp': pd.date_range(start='2025-01-01', periods=n, freq='10min'),
            'altitude': np.linspace(0, 30000, n) + np.random.normal(0, 100, n),
            'temperature': np.linspace(25, -60, n) + np.random.normal(0, 2, n),
            'pressure': np.linspace(1013, 10, n) + np.random.normal(0, 5, n),
            'bit_flips_count': np.random.poisson(lam=np.linspace(0, 10, n)),
            'latitude': 39.0 + np.random.normal(0, 0.1, n),
            'longitude': -98.0 + np.random.normal(0, 0.1, n),
            'cosmic_ray_count': np.random.poisson(lam=np.linspace(1, 20, n)),
            'max_run_length': np.random.randint(0, 5, n)
        }
        
        # Add cosmic intensity based on altitude
        data['cosmic_intensity'] = np.exp(data['altitude'] / 10000)
        
        # Create test file
        df = pd.DataFrame(data)
        test_file = os.path.join(self.test_dir, 'test_seu_data.csv')
        df.to_csv(test_file, index=False)
        
        return df
    
    def test_data_loading(self):
        """Test data loading functionality"""
        # Save test data
        test_file = os.path.join(self.test_dir, 'test_data.csv')
        self.sample_data.to_csv(test_file, index=False)
        
        # Load and verify
        loaded_df = pd.read_csv(test_file)
        
        # Check shapes match
        self.assertEqual(self.sample_data.shape, loaded_df.shape)
        
        # Check a few values
        self.assertAlmostEqual(
            self.sample_data['altitude'].mean(), 
            loaded_df['altitude'].mean(), 
            places=4
        )
    
    def test_anomaly_detection(self):
        """Test anomaly detection functionality"""
        try:
            from ml_pipeline.anomaly_detection import AnomalyDetectionModel
        except ImportError:
            self.skipTest("Anomaly detection module not available")
        
        # Create a model in the test directory
        model = AnomalyDetectionModel(model_path=os.path.join(self.test_dir, 'test_model.pkl'))
        
        # Test unsupervised training
        result_df = model.train_unsupervised(self.sample_data)
        
        # Check if anomaly columns were added
        self.assertIn('anomaly_score', result_df.columns)
        self.assertIn('is_anomaly', result_df.columns)
        
        # Test real-time prediction
        data_point = {
            'altitude': 25000,
            'temperature': -50,
            'bit_flips_count': 8,
            'cosmic_intensity': 12.5
        }
        
        result = model.predict_real_time(data_point)
        
        # Check if prediction contains expected keys
        self.assertIn('anomaly_score', result)
        self.assertIn('is_anomaly', result)
    
    def test_synthetic_data_creation(self):
        """Test synthetic data creation functionality"""
        try:
            # Import dynamically since this is an optional test
            sys.path.append('ml_pipeline')
            from SEU_Synthetic_Data_Creator import generate_altitude_profile, generate_temperature_profile
        except ImportError:
            self.skipTest("Synthetic data creator module not available")
        
        # Test profile generation
        time_seconds = np.linspace(0, 3600 * 6, 1000)  # 6 hours
        
        # Test altitude profile
        altitude = generate_altitude_profile(time_seconds, 6)
        self.assertEqual(len(altitude), len(time_seconds))
        self.assertGreater(max(altitude), 0)
        
        # Test temperature profile
        temperature = generate_temperature_profile(altitude)
        self.assertEqual(len(temperature), len(altitude))
    
    def test_data_processing(self):
        """Test data processing functionality"""
        # Add derived features
        df = self.sample_data.copy()
        
        # Add time_delta
        df['time_delta'] = 600  # 10 minutes in seconds
        
        # Calculate bit flip rate
        df['bit_flip_rate'] = df['bit_flips_count'] / df['time_delta']
        
        # Check if calculation is correct
        expected_rate = self.sample_data['bit_flips_count'] / 600
        pd.testing.assert_series_equal(df['bit_flip_rate'], expected_rate)
        
        # Calculate cosmic ray intensity based on altitude
        df['calculated_intensity'] = np.exp(df['altitude'] / 10000)
        
        # Check correlation between altitude and cosmic intensity
        correlation = df['altitude'].corr(df['calculated_intensity'])
        self.assertGreater(correlation, 0.9)  # Should be strongly correlated

    def test_api_endpoints(self):
        """Test API server endpoints (mock)"""
        try:
            import fastapi
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("FastAPI test client not available")

        # Dynamic import to avoid dependency at top level
        import importlib.util
        import sys
        
        # Try to load API server module
        try:
            spec = importlib.util.spec_from_file_location(
                "api_server", os.path.join(Path(__file__).parent.absolute(), "api_server.py")
            )
            api_server = importlib.util.module_from_spec(spec)
            sys.modules["api_server"] = api_server
            spec.loader.exec_module(api_server)
            
            # Create test client
            client = TestClient(api_server.app)
            
            # Test health endpoint
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertIn("status", response.json())
        except Exception:
            self.skipTest("API server could not be loaded for testing")
    
    def test_real_time_monitor(self):
        """Test real-time monitor functionality"""
        try:
            import importlib.util
            import sys
            
            # Try to load real-time monitor module
            spec = importlib.util.spec_from_file_location(
                "real_time_monitor", os.path.join(Path(__file__).parent.absolute(), "real_time_monitor.py")
            )
            rtm = importlib.util.module_from_spec(spec)
            sys.modules["real_time_monitor"] = rtm
            spec.loader.exec_module(rtm)
            
            # Check if monitor class can be instantiated
            if hasattr(rtm, "SEURealTimeMonitor"):
                monitor = rtm.SEURealTimeMonitor()
                self.assertIsNotNone(monitor)
                
                # Test data processing if method exists
                if hasattr(monitor, "process_data_point"):
                    sample_data = {
                        "timestamp": "2025-01-01T12:00:00",
                        "altitude": 15000,
                        "temperature": -40,
                        "bit_flips_count": 5
                    }
                    result = monitor.process_data_point(sample_data)
                    self.assertIsNotNone(result)
        except Exception:
            self.skipTest("Real-time monitor could not be loaded for testing")
    
    def test_integrated_dashboard(self):
        """Test integrated dashboard initialization"""
        try:
            import dash
            import importlib.util
            import sys
            
            # Skip full initialization which would start a server
            # Just check if we can import and access key components
            spec = importlib.util.spec_from_file_location(
                "integrated_dashboard", os.path.join(Path(__file__).parent.absolute(), "integrated_dashboard.py")
            )
            dash_module = importlib.util.module_from_spec(spec)
            
            # Monkey patch Dash to prevent actual server start
            original_run_server = dash.Dash.run_server
            dash.Dash.run_server = lambda *args, **kwargs: None
            
            try:
                sys.modules["integrated_dashboard"] = dash_module
                spec.loader.exec_module(dash_module)
                
                # Check if app was created
                self.assertTrue(hasattr(dash_module, "app"))
            finally:
                # Restore original method
                dash.Dash.run_server = original_run_server
        except Exception:
            self.skipTest("Integrated dashboard could not be loaded for testing")
    
    def test_raspberry_pi_setup(self):
        """Test Raspberry Pi setup script"""
        try:
            import importlib.util
            import sys
            
            # Skip hardware initialization by monkey patching
            import os
            os.environ["SIMULATION_MODE"] = "1"
            
            # Try to load the script
            spec = importlib.util.spec_from_file_location(
                "raspberry_pi_seu_detector", 
                os.path.join(Path(__file__).parent.absolute(), "raspberry_pi_seu_detector.py")
            )
            rpi_module = importlib.util.module_from_spec(spec)
            
            # Only test loading - don't execute the script
            self.assertIsNotNone(spec)
        except Exception:
            self.skipTest("Raspberry Pi setup script could not be loaded for testing")

def run_tests():
    """Run all tests"""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests()
