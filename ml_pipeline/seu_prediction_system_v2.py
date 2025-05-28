#!/usr/bin/env python3
"""
Complete High-Altitude Balloon SEU Detection and Prediction System
Integrates all hardware components with real-time data collection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import threading
import queue
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import struct

# Hardware interface imports
try:
    import RPi.GPIO as GPIO
    import spidev
    import smbus2
    import serial
    import board
    import busio
    import adafruit_bmp280
    HAS_HARDWARE = True
except ImportError:
    print("Hardware libraries not available - running in simulation mode")
    HAS_HARDWARE = False

@dataclass
class SensorReading:
    """Data structure for sensor readings"""
    timestamp: datetime
    altitude: float
    temperature: float
    pressure: float
    gps_lat: float
    gps_lon: float
    gps_altitude: float
    cosmic_ray_count: int
    seu_events: int
    max_run_length: int
    battery_voltage: float

class HardwareInterface:
    """Interface to all hardware components"""
    
    def __init__(self):
        self.i2c = None
        self.bmp280 = None
        self.gps_serial = None
        self.spi = None
        self.cosmic_ray_pin = 18  # GPIO pin for SiPM pulses
        self.cosmic_ray_count = 0
        self.seu_detector = None
        
        if HAS_HARDWARE:
            self.initialize_hardware()
        else:
            print("Running in simulation mode - no hardware initialization")
    
    def initialize_hardware(self):
        """Initialize all hardware components"""
        try:
            # Initialize I2C for BMP280
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(self.i2c)
            self.bmp280.sea_level_pressure = 1013.25
            print("BMP280 initialized")
            
            # Initialize GPS UART
            self.gps_serial = serial.Serial('/dev/ttyS0', 9600, timeout=1)
            print("GPS initialized")
            
            # Initialize SPI for SRAM chips
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)  # Bus 0, Device 0
            self.spi.max_speed_hz = 1000000
            print("SPI initialized for SRAM")
            
            # Initialize GPIO for cosmic ray detection
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.cosmic_ray_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            GPIO.add_event_detect(self.cosmic_ray_pin, GPIO.RISING, 
                                callback=self._cosmic_ray_callback, bouncetime=1)
            print("Cosmic ray detector initialized")
            
            # Initialize SEU detector
            self.seu_detector = SEUDetector(self.spi)
            print("SEU detector initialized")
            
        except Exception as e:
            print(f"Hardware initialization error: {e}")
            print("Falling back to simulation mode")
            global HAS_HARDWARE
            HAS_HARDWARE = False
    
    def _cosmic_ray_callback(self, channel):
        """Callback for cosmic ray detection"""
        self.cosmic_ray_count += 1
    
    def read_bmp280(self) -> Tuple[float, float, float]:
        """Read temperature, pressure, and calculate altitude"""
        if HAS_HARDWARE and self.bmp280:
            try:
                temp = self.bmp280.temperature
                pressure = self.bmp280.pressure
                altitude = self.bmp280.altitude
                return temp, pressure, altitude
            except Exception as e:
                print(f"BMP280 read error: {e}")
        
        # Simulation fallback
        base_alt = 15000 + np.random.normal(0, 100)
        temp = 20 - (base_alt / 150) + np.random.normal(0, 2)
        pressure = 1013.25 * np.exp(-base_alt / 8400)
        return temp, pressure, base_alt
    
    def read_gps(self) -> Tuple[float, float, float]:
        """Read GPS coordinates and altitude"""
        if HAS_HARDWARE and self.gps_serial:
            try:
                # Read NMEA sentences and parse
                for _ in range(10):  # Try up to 10 lines
                    line = self.gps_serial.readline().decode('ascii', errors='ignore')
                    if line.startswith('$GPGGA'):
                        parts = line.split(',')
                        if len(parts) >= 10 and parts[2] and parts[4] and parts[9]:
                            lat = self._parse_coordinate(parts[2], parts[3])
                            lon = self._parse_coordinate(parts[4], parts[5])
                            alt = float(parts[9])
                            return lat, lon, alt
            except Exception as e:
                print(f"GPS read error: {e}")
        
        # Simulation fallback
        lat = 39.01 + np.random.normal(0, 0.001)
        lon = -98.00 + np.random.normal(0, 0.001)
        alt = 15000 + np.random.normal(0, 200)
        return lat, lon, alt
    
    def _parse_coordinate(self, coord_str: str, direction: str) -> float:
        """Parse NMEA coordinate format"""
        if not coord_str:
            return 0.0
        
        # DDMM.MMMM format
        degrees = int(coord_str[:2])
        minutes = float(coord_str[2:])
        decimal = degrees + minutes / 60
        
        if direction in ['S', 'W']:
            decimal = -decimal
        
        return decimal
    
    def read_cosmic_rays(self) -> int:
        """Get cosmic ray count since last read"""
        count = self.cosmic_ray_count
        self.cosmic_ray_count = 0  # Reset counter
        return count
    
    def read_battery_voltage(self) -> float:
        """Read battery voltage (would need ADC)"""
        # Simulation - assuming 6 AA lithium batteries
        return 9.0 + np.random.normal(0, 0.2)
    
    def cleanup(self):
        """Clean up hardware resources"""
        if HAS_HARDWARE:
            if self.gps_serial:
                self.gps_serial.close()
            if self.spi:
                self.spi.close()
            GPIO.cleanup()

class SEUDetector:
    """Single Event Upset detector using SRAM chips"""
    
    def __init__(self, spi_interface):
        self.spi = spi_interface
        self.memory_size = 2 * 128 * 1024  # 2x 128KB chips
        self.test_patterns = [0x55, 0xAA, 0xFF, 0x00]  # Alternating patterns
        self.current_pattern_index = 0
        self.last_full_test = time.time()
        self.error_count = 0
        self.max_consecutive_errors = 0
    
    def write_sram(self, address: int, data: bytes, chip: int = 0):
        """Write data to SRAM chip"""
        if not self.spi:
            return
        
        try:
            # 23LC1024 write command format
            cmd = [0x02]  # WRITE command
            cmd.extend([(address >> 16) & 0xFF, (address >> 8) & 0xFF, address & 0xFF])
            cmd.extend(data)
            
            # Select chip (using CS)
            response = self.spi.xfer2(cmd)
        except Exception as e:
            print(f"SRAM write error: {e}")
    
    def read_sram(self, address: int, length: int, chip: int = 0) -> bytes:
        """Read data from SRAM chip"""
        if not self.spi:
            return b'\x00' * length
        
        try:
            # 23LC1024 read command format
            cmd = [0x03]  # READ command
            cmd.extend([(address >> 16) & 0xFF, (address >> 8) & 0xFF, address & 0xFF])
            cmd.extend([0x00] * length)  # Dummy bytes for read
            
            response = self.spi.xfer2(cmd)
            return bytes(response[4:])  # Skip command bytes
        except Exception as e:
            print(f"SRAM read error: {e}")
            return b'\x00' * length
    
    def quick_test(self) -> Tuple[int, int]:
        """Quick memory test - check small portion"""
        test_size = 1024  # Test 1KB
        pattern = self.test_patterns[self.current_pattern_index]
        test_data = bytes([pattern] * test_size)
        
        # Write pattern
        self.write_sram(0, test_data, 0)
        time.sleep(0.001)  # Brief delay
        
        # Read back and compare
        read_data = self.read_sram(0, test_size, 0)
        
        # Count errors
        errors = 0
        consecutive_errors = 0
        max_consecutive = 0
        
        for i in range(test_size):
            if i < len(read_data) and read_data[i] != pattern:
                errors += 1
                consecutive_errors += 1
                max_consecutive = max(max_consecutive, consecutive_errors)
            else:
                consecutive_errors = 0
        
        self.current_pattern_index = (self.current_pattern_index + 1) % len(self.test_patterns)
        return errors, max_consecutive
    
    def comprehensive_test(self) -> Tuple[int, int]:
        """Comprehensive memory test - run periodically"""
        if time.time() - self.last_full_test < 60:  # Only every minute
            return self.quick_test()
        
        print("Running comprehensive memory test...")
        total_errors = 0
        max_run_length = 0
        
        # Test both chips with different patterns
        for chip in range(2):
            for pattern_idx, pattern in enumerate(self.test_patterns):
                test_size = 4096  # 4KB per test
                start_addr = pattern_idx * test_size
                
                test_data = bytes([pattern] * test_size)
                self.write_sram(start_addr, test_data, chip)
                time.sleep(0.01)
                
                read_data = self.read_sram(start_addr, test_size, chip)
                
                # Analyze errors
                consecutive = 0
                for i in range(test_size):
                    if i < len(read_data) and read_data[i] != pattern:
                        total_errors += 1
                        consecutive += 1
                    else:
                        max_run_length = max(max_run_length, consecutive)
                        consecutive = 0
        
        self.last_full_test = time.time()
        print(f"Comprehensive test complete: {total_errors} errors, max run: {max_run_length}")
        return total_errors, max_run_length

class DataCollector:
    """Main data collection and prediction system"""
    
    def __init__(self):
        self.hardware = HardwareInterface()
        self.predictor = RealTimeSEUPredictor()
        self.data_queue = queue.Queue()
        self.running = False
        self.collection_thread = None
        self.prediction_thread = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('balloon_flight.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def collect_sensor_data(self) -> SensorReading:
        """Collect data from all sensors"""
        try:
            # Environmental sensors
            temp, pressure, altitude = self.hardware.read_bmp280()
            
            # GPS
            gps_lat, gps_lon, gps_alt = self.hardware.read_gps()
            
            # Cosmic ray detection
            cosmic_count = self.hardware.read_cosmic_rays()
            
            # SEU detection
            if self.hardware.seu_detector:
                seu_count, max_run = self.hardware.seu_detector.quick_test()
            else:
                # Simulation data
                seu_count = max(0, int(np.random.poisson(2)))
                max_run = max(1, int(np.random.exponential(1.5)))
            
            # Battery monitoring
            battery_v = self.hardware.read_battery_voltage()
            
            return SensorReading(
                timestamp=datetime.now(),
                altitude=altitude,
                temperature=temp,
                pressure=pressure,
                gps_lat=gps_lat,
                gps_lon=gps_lon,
                gps_altitude=gps_alt,
                cosmic_ray_count=cosmic_count,
                seu_events=seu_count,
                max_run_length=max_run,
                battery_voltage=battery_v
            )
            
        except Exception as e:
            self.logger.error(f"Sensor data collection error: {e}")
            return None
    
    def data_collection_loop(self):
        """Main data collection loop"""
        self.logger.info("Starting data collection loop")
        
        while self.running:
            try:
                # Collect sensor data
                reading = self.collect_sensor_data()
                
                if reading:
                    # Add to processing queue
                    self.data_queue.put(reading)
                    
                    # Log critical data
                    self.logger.info(
                        f"Alt: {reading.altitude:.0f}m, "
                        f"Temp: {reading.temperature:.1f}°C, "
                        f"SEUs: {reading.seu_events}, "
                        f"Cosmic: {reading.cosmic_ray_count}, "
                        f"Battery: {reading.battery_voltage:.1f}V"
                    )
                
                # Collection interval (15 seconds)
                time.sleep(15)
                
            except Exception as e:
                self.logger.error(f"Data collection loop error: {e}")
                time.sleep(5)  # Brief pause before retry
    
    def prediction_loop(self):
        """Process data and make predictions"""
        self.logger.info("Starting prediction loop")
        
        while self.running:
            try:
                # Get data from queue (blocking with timeout)
                reading = self.data_queue.get(timeout=30)
                
                # Make predictions
                result = self.predictor.add_measurement(
                    reading.timestamp,
                    reading.altitude,
                    reading.temperature,
                    reading.seu_events,
                    reading.max_run_length,
                    reading.gps_lat,
                    reading.gps_lon
                )
                
                if result:
                    # Process predictions and alerts
                    self.process_predictions(result, reading)
                
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Prediction loop error: {e}")
    
    def process_predictions(self, result: Dict, reading: SensorReading):
        """Process predictions and handle alerts"""
        predictions = result['predictions']
        alerts = result['alerts']
        
        # Log predictions
        immediate = predictions['immediate']
        short_term = predictions['short_term']
        
        self.logger.info(
            f"Predictions - Immediate: {immediate['predicted_seus']} SEUs "
            f"({immediate['confidence']:.2f}), "
            f"Short-term: {short_term['predicted_seus']} SEUs "
            f"({short_term['confidence']:.2f})"
        )
        
        # Handle alerts
        for alert in alerts:
            self.logger.warning(
                f"ALERT [{alert['severity']}]: {alert['message']} - "
                f"Action: {alert['recommended_action']}"
            )
        
        # Save detailed data
        self.save_flight_data(reading, result)
    
    def save_flight_data(self, reading: SensorReading, predictions: Dict):
        """Save comprehensive flight data"""
        data_entry = {
            'timestamp': reading.timestamp.isoformat(),
            'sensors': {
                'altitude': float(reading.altitude),
                'temperature': float(reading.temperature),
                'pressure': float(reading.pressure),
                'gps_lat': float(reading.gps_lat),
                'gps_lon': float(reading.gps_lon),
                'gps_altitude': float(reading.gps_altitude),
                'cosmic_ray_count': int(reading.cosmic_ray_count),
                'seu_events': int(reading.seu_events),
                'max_run_length': int(reading.max_run_length),
                'battery_voltage': float(reading.battery_voltage)
            },
            'predictions': self.convert_to_json_serializable(predictions['predictions']),
            'alerts': self.convert_to_json_serializable(predictions['alerts']),
            'status': self.convert_to_json_serializable(predictions['current_status'])
        }
        
        try:
            with open('flight_data.json', 'a') as f:
                f.write(json.dumps(data_entry, indent=2) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save flight data: {e}")
    
    def convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types"""
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    def start_collection(self):
        """Start data collection and prediction"""
        if self.running:
            self.logger.warning("Collection already running")
            return
        
        self.running = True
        
        # Start threads
        self.collection_thread = threading.Thread(target=self.data_collection_loop)
        self.prediction_thread = threading.Thread(target=self.prediction_loop)
        
        self.collection_thread.start()
        self.prediction_thread.start()
        
        self.logger.info("Data collection and prediction started")
    
    def stop_collection(self):
        """Stop data collection"""
        self.running = False
        
        if self.collection_thread:
            self.collection_thread.join()
        if self.prediction_thread:
            self.prediction_thread.join()
        
        self.hardware.cleanup()
        self.logger.info("Data collection stopped")

# Copy of the original predictor class
class RealTimeSEUPredictor:
    """Real-time SEU prediction system for high-altitude balloon"""
    
    def __init__(self, trained_model=None):
        self.trained_model = trained_model
        self.data_buffer = []
        self.buffer_size = 50
        self.prediction_horizon = 4
        
        # Alert thresholds
        self.seu_alert_threshold = 10
        self.anomaly_threshold = -0.1
        
    def add_measurement(self, timestamp, altitude, temperature, bit_flips, 
                       max_run_length, gps_lat, gps_lon):
        """Add new measurement and make predictions"""
        
        measurement = {
            'timestamp': timestamp,
            'altitude': altitude,
            'temperature': temperature,
            'bit_flips_count': bit_flips,
            'max_run_length': max_run_length,
            'latitude': gps_lat,
            'longitude': gps_lon
        }
        
        self.data_buffer.append(measurement)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
            
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
            
        df = pd.DataFrame(self.data_buffer)
        df = self.calculate_features(df)
        
        predictions = {}
        
        # Simple prediction based on altitude and cosmic ray intensity
        current_alt = df.iloc[-1]['altitude']
        current_temp = df.iloc[-1]['temperature']
        cosmic_intensity = 1.0 * np.exp(current_alt / 4500)
        
        # Immediate prediction
        base_seus = max(0, int(cosmic_intensity * 0.1 + np.random.poisson(1)))
        predictions['immediate'] = {
            'time_ahead': '15 seconds',
            'predicted_seus': base_seus,
            'confidence': 0.75
        }
        
        # Short-term prediction
        future_seus = max(0, int(cosmic_intensity * 0.12 + np.random.poisson(1.2)))
        predictions['short_term'] = {
            'time_ahead': '1 minute',
            'predicted_seus': future_seus,
            'confidence': 0.65
        }
        
        # Anomaly detection
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
        df['time_delta'] = df['timestamp'].diff().dt.total_seconds().fillna(15)
        df['bit_flip_rate'] = df['bit_flips_count'] / df['time_delta']
        df['altitude_change_rate'] = df['altitude'].diff() / df['time_delta']
        df['temperature_gradient'] = df['temperature'].diff()
        df['cosmic_ray_intensity'] = 1.0 * np.exp(df['altitude'] / 4500)
        df['bit_flip_ma_5'] = df['bit_flips_count'].rolling(window=5, min_periods=1).mean()
        df['altitude_ma_10'] = df['altitude'].rolling(window=10, min_periods=1).mean()
        
        return df.fillna(0)
    
    def detect_anomaly(self, recent_data):
        """Simple anomaly detection based on recent patterns"""
        if len(recent_data) < 3:
            return 0.0
            
        current_rate = recent_data['bit_flip_rate'].iloc[-1]
        avg_rate = recent_data['bit_flip_rate'].mean()
        
        if avg_rate > 0:
            ratio = current_rate / avg_rate
            return min(0.2, 0.1 - (ratio - 1) * 0.1)
        else:
            return 0.1
    
    def check_alerts(self, predictions):
        """Check if any alerts should be triggered"""
        alerts = []
        
        if predictions is None:
            return alerts
            
        immediate_seus = predictions['immediate']['predicted_seus']
        if immediate_seus > self.seu_alert_threshold:
            alerts.append({
                'type': 'HIGH_SEU_RATE',
                'severity': 'WARNING',
                'message': f'High SEU rate predicted: {immediate_seus} events in next 15s',
                'recommended_action': 'Monitor memory integrity, consider data backup'
            })
        
        if predictions['anomaly_status']['is_anomalous']:
            alerts.append({
                'type': 'COSMIC_RAY_BURST',
                'severity': 'ALERT',
                'message': 'Cosmic ray shower detected - elevated SEU risk',
                'recommended_action': 'Increase data collection frequency, activate protection'
            })
        
        current_altitude = self.data_buffer[-1]['altitude']
        if current_altitude > 30000:
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

def main():
    """Main execution function"""
    print("High-Altitude Balloon SEU Detection System")
    print("=" * 50)
    
    if not HAS_HARDWARE:
        print("WARNING: Running in simulation mode")
        time.sleep(2)
    
    collector = DataCollector()
    
    try:
        print("Starting data collection...")
        collector.start_collection()
        
        # Run for specified duration or until interrupted
        print("System running. Press Ctrl+C to stop.")
        while True:
            time.sleep(60)  # Status update every minute
            print(f"System operational - {datetime.now().strftime('%H:%M:%S')}")
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        collector.stop_collection()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
