#!/usr/bin/env python3
"""
SEU Detector - Raspberry Pi Zero W Data Collection Script
This script runs on the Raspberry Pi Zero W and collects data from:
1. SRAM chips via SPI
2. BMP280 temperature/pressure sensor via I2C
3. GPS module via UART
4. Cosmic ray counter via GPIO
"""

import os
import time
import json
import logging
import threading
import queue
from datetime import datetime
import struct
import signal
import sys
import argparse
import traceback
import shutil
import gzip
import csv
import io
from typing import Dict, List, Optional, Tuple, Union, Any, TypedDict, TextIO
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, field
from collections import deque

# Hardware-dependent libraries with fallbacks
try:
    # Raspberry Pi hardware interfaces
    import RPi.GPIO as GPIO
    import spidev
    import smbus2 as smbus  # smbus2 is more robust than smbus
    import serial
    
    # Adafruit libraries for sensors
    import board
    import busio
    import adafruit_bmp280
    import adafruit_gps
    
    # Communication libraries
    import paho.mqtt.client as mqtt
    
    # Optional hardware monitoring
    try:
        import psutil  # For system resource monitoring
    except ImportError:
        psutil = None
        print("Warning: psutil not available - reduced system monitoring capabilities")
    
    HAS_HARDWARE = True
except ImportError as e:
    print(f"Warning: Hardware library not available ({str(e)})")
    print("Running in simulation mode - hardware libraries not available")
    HAS_HARDWARE = False
    
    # Simulation dependencies
    import random
    import numpy as np
    from datetime import timedelta
    
    # Mock hardware interfaces for simulation
    class MockGPIO:
        """Mock GPIO implementation for simulation mode"""
        BOARD = "BOARD"
        BCM = "BCM"
        IN = "IN"
        OUT = "OUT"
        PUD_UP = "PUD_UP"
        PUD_DOWN = "PUD_DOWN"
        RISING = "RISING"
        FALLING = "FALLING"
        BOTH = "BOTH"
        
        @classmethod
        def setmode(cls, mode):
            pass
            
        @classmethod
        def setup(cls, pin, direction, pull_up_down=None):
            pass
            
        @classmethod
        def add_event_detect(cls, pin, edge, callback=None, bouncetime=None):
            pass
            
        @classmethod
        def output(cls, pin, value):
            pass
            
        @classmethod
        def input(cls, pin):
            return random.choice([0, 1])
            
        @classmethod
        def cleanup(cls):
            pass
    
    # Use mock interfaces when hardware is not available
    GPIO = MockGPIO()
    psutil = None
    
    # Optional simulation visualization
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("Warning: Matplotlib not available - visualization disabled")

# Network communication libraries
try:
    import requests
    import urllib3
    HAS_REQUESTS = True
except ImportError:
    print("Warning: Requests library not available - remote data transmission disabled")
    HAS_REQUESTS = False

# Data compression libraries
try:
    import zlib
    import bz2
    import lzma
    HAS_COMPRESSION = True
except ImportError:
    print("Warning: Advanced compression libraries not available - using gzip only")
    HAS_COMPRESSION = False

# Set up constants
VERSION = "1.2.0"
DEFAULT_LOG_PATH = "seu_detector.log"
DEFAULT_DATA_DIR = "data"
DEFAULT_CONFIG_PATH = "seu_detector_config.json"

# Check if environment variable for simulation is set
if os.environ.get("SIMULATION_MODE", "").lower() in ("1", "true", "yes"):
    HAS_HARDWARE = False
    print("Simulation mode enabled via environment variable")

# Set up logging
log_file = os.environ.get("SEU_LOG_FILE", DEFAULT_LOG_PATH)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SEU-Detector")

# Log program start with version
logger.info(f"SEU Detector v{VERSION} starting {'(SIMULATION MODE)' if not HAS_HARDWARE else ''}")

# Configuration
DEFAULT_CONFIG = {
    # Timing parameters
    "sram_check_interval": 15,       # seconds between SRAM checks
    "sensor_read_interval": 2,       # seconds between sensor readings
    "log_interval": 1,               # seconds between logging data
    
    # Hardware settings
    "test_pattern": [0x55, 0xAA],    # SRAM test patterns (alternating bits)
    "spi_speed_hz": 1000000,         # SPI speed (1MHz)
    "spi_bus": 0,                    # SPI bus
    "spi_device": 0,                 # SPI device
    "i2c_bus": 1,                    # I2C bus
    "cosmic_ray_pin": 18,            # GPIO pin for cosmic ray counter
    "gps_port": "/dev/ttyS0",        # Serial port for GPS
    "gps_baud": 9600,                # GPS baud rate
    
    # Data transmission
    "mqtt_broker": "localhost",      # MQTT broker address
    "mqtt_port": 1883,               # MQTT broker port
    "mqtt_topic": "seu_detector",    # MQTT topic for publishing
    "mqtt_enabled": False,           # Enable MQTT publishing
    "remote_api_url": "",            # API endpoint for data transmission
    "remote_api_enabled": False,     # Enable remote API transmission
    
    # Data storage
    "serial_output": True,           # Output to serial for external reading
    "data_storage_path": DEFAULT_DATA_DIR, # Directory for storing data
    "save_daily_files": True,        # Create new file each day
    "compress_old_files": True,      # Compress files older than 1 day
    
    # Operational settings
    "simulation_mode": not HAS_HARDWARE,  # Auto simulation if hardware not available
    "visualize_simulation": False,   # Create live plot in simulation mode
    "battery_saving": False,         # Enable battery saving features
    "restart_on_error": True,        # Auto restart on critical error
}

# Global variables
running = True
data_queue = queue.Queue()
mqtt_client = None

class SEUDetector:
    """Main class for SEU detection hardware interface"""
    
    def __init__(self, config: Dict):
        """Initialize the SEU detector with the given configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        
        # Hardware interfaces
        self.spi = None
        self.i2c = None
        self.bmp280 = None
        self.gps_serial = None
        self.cosmic_ray_pin = config.get("cosmic_ray_pin", 18)
        
        # Counters and timers
        self.cosmic_ray_count = 0
        self.last_sram_check = time.time()
        self.last_sensor_read = time.time()
        self.last_log_time = time.time()
        self.current_pattern = 0
        self.start_time = time.time()
        self.error_count = 0
        
        # Simulation variables
        if config["simulation_mode"]:
            self.sim_start_time = time.time()
            self.sim_plot = None
            self.sim_figure = None
            self.sim_data_points = []
        
        # Initialize data structure
        self.data = {
            "timestamp": datetime.now().isoformat(),
            "altitude": 0,
            "temperature": 0,
            "pressure": 0,
            "latitude": 0,
            "longitude": 0,
            "gps_altitude": 0,
            "bit_flips_count": 0,
            "max_run_length": 0,
            "cosmic_ray_count": 0,
            "battery_voltage": 0,
            "sram_regions": [0, 0, 0, 0],  # 4 SRAM regions
            "device_id": self._get_device_id(),
            "uptime": 0,
            "version": VERSION
        }
        
        # Create data directory if needed
        try:
            os.makedirs(self.config["data_storage_path"], exist_ok=True)
            logger.info(f"Data will be stored in: {os.path.abspath(self.config['data_storage_path'])}")
        except Exception as e:
            logger.error(f"Failed to create data directory: {str(e)}")
            # Fall back to current directory
            self.config["data_storage_path"] = "."
            logger.info("Using current directory for data storage")
        
        # Initialize hardware or simulation
        if not config["simulation_mode"]:
            self.initialize_hardware()
        else:
            logger.info("Running in simulation mode")
            # Initialize visualization if enabled
            if config.get("visualize_simulation", False) and HAS_MATPLOTLIB:
                self._initialize_simulation_visualization()
    
    def _get_device_id(self) -> str:
        """Generate a unique device ID based on hardware or software information"""
        if HAS_HARDWARE:
            try:
                # Try to get Raspberry Pi serial number
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('Serial'):
                            return line.split(':')[1].strip()
                
                # Fallback to MAC address
                from uuid import getnode
                mac = getnode()
                return f"RPi-{mac:012x}"
            except Exception:
                pass
        
        # Software-based fallback ID
        import socket
        import hashlib
        hostname = socket.gethostname()
        return f"SEU-{hashlib.md5(hostname.encode()).hexdigest()[:8]}"
    
    def _initialize_simulation_visualization(self):
        """Initialize matplotlib for real-time visualization in simulation mode"""
        if not HAS_MATPLOTLIB:
            return
        
        try:
            plt.ion()  # Interactive mode
            self.sim_figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Setup altitude and temperature plot
            ax1.set_title('SEU Detector Simulation')
            ax1.set_ylabel('Altitude (m) / Temperature (°C)')
            ax1.grid(True)
            
            # Setup SEU events plot
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('SEU Events / Cosmic Rays')
            ax2.grid(True)
            
            # Initialize empty data
            self.sim_altitude_line, = ax1.plot([], [], 'b-', label='Altitude')
            self.sim_temp_line, = ax1.plot([], [], 'r-', label='Temperature')
            self.sim_seu_line, = ax2.plot([], [], 'g-', label='SEU Events')
            self.sim_cosmic_line, = ax2.plot([], [], 'm-', label='Cosmic Rays')
            
            ax1.legend()
            ax2.legend()
            
            plt.tight_layout()
            plt.show(block=False)
            
            logger.info("Simulation visualization initialized")
        except Exception as e:
            logger.error(f"Failed to initialize visualization: {str(e)}")
            self.sim_figure = None
    
    def initialize_hardware(self):
        """Initialize all hardware components"""
        try:
            # Initialize SPI for SRAM communication
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)  # Bus 0, Device 0
            self.spi.max_speed_hz = self.config["spi_speed_hz"]
            logger.info("SPI initialized for SRAM")
            
            # Initialize I2C for BMP280 sensor
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(self.i2c)
            self.bmp280.sea_level_pressure = 1013.25
            logger.info("BMP280 initialized")
            
            # Initialize GPS serial connection
            self.gps_serial = serial.Serial('/dev/ttyS0', 9600, timeout=1)
            logger.info("GPS initialized")
            
            # Initialize GPIO for cosmic ray detection
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.cosmic_ray_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            GPIO.add_event_detect(self.cosmic_ray_pin, GPIO.RISING, 
                                callback=self._cosmic_ray_callback, bouncetime=1)
            logger.info("Cosmic ray detector initialized")
            
            # Initial SRAM pattern write
            self._initialize_sram_patterns()
            
            logger.info("All hardware initialized successfully")
        except Exception as e:
            logger.error(f"Hardware initialization error: {str(e)}")
            logger.info("Falling back to simulation mode")
            self.config["simulation_mode"] = True
    
    def _cosmic_ray_callback(self, channel):
        """Callback function for cosmic ray counter"""
        self.cosmic_ray_count += 1
    
    def _initialize_sram_patterns(self):
        """Initialize SRAM with test patterns"""
        if self.config["simulation_mode"] or not self.spi:
            return
        
        try:
            logger.info("Initializing SRAM with test patterns...")
            pattern = self.config["test_pattern"][self.current_pattern]
            
            # Define SRAM regions (4 regions for testing different parts)
            sram_size = 128 * 1024  # 128KB chip
            region_size = sram_size // 4
            
            for region in range(4):
                start_addr = region * region_size
                end_addr = (region + 1) * region_size
                
                for addr in range(start_addr, end_addr):
                    self.write_sram(addr, [pattern])
            
            logger.info(f"SRAM initialized with pattern 0x{pattern:02X}")
        except Exception as e:
            logger.error(f"SRAM initialization error: {str(e)}")
    
    def write_sram(self, address, data):
        """Write data to SRAM"""
        if self.config["simulation_mode"] or not self.spi:
            return
        
        try:
            # Format SRAM write command
            cmd = [0x02]  # WRITE command
            cmd.extend([(address >> 16) & 0xFF, (address >> 8) & 0xFF, address & 0xFF])
            cmd.extend(data)
            self.spi.xfer2(cmd)
        except Exception as e:
            logger.error(f"SRAM write error: {str(e)}")
    
    def read_sram(self, address, nbytes=1):
        """Read data from SRAM"""
        if self.config["simulation_mode"] or not self.spi:
            return [0] * nbytes
        
        try:
            # Format SRAM read command
            cmd = [0x03]  # READ command
            cmd.extend([(address >> 16) & 0xFF, (address >> 8) & 0xFF, address & 0xFF])
            cmd.extend([0] * nbytes)  # Dummy bytes to receive data
            result = self.spi.xfer2(cmd)
            return result[4:]  # Skip the 4-byte command header
        except Exception as e:
            logger.error(f"SRAM read error: {str(e)}")
            return [0] * nbytes
    
    def check_sram_errors(self):
        """Check SRAM for errors against expected pattern"""
        if self.config["simulation_mode"]:
            # Simulate SEU events based on altitude
            altitude = self.data["altitude"]
            if altitude < 100:  # Ground level
                self.data["bit_flips_count"] = 0
                self.data["max_run_length"] = 0
                self.data["sram_regions"] = [0, 0, 0, 0]
            else:
                # Higher altitude = more SEUs
                base_probability = min(1.0, altitude / 50000)
                seu_count = np.random.poisson(base_probability * 10)
                self.data["bit_flips_count"] = seu_count
                
                if seu_count > 0:
                    self.data["max_run_length"] = min(seu_count, np.random.poisson(2) + 1)
                    # Distribute SEUs across regions
                    self.data["sram_regions"] = [
                        np.random.binomial(seu_count, 0.25) for _ in range(4)
                    ]
                else:
                    self.data["max_run_length"] = 0
                    self.data["sram_regions"] = [0, 0, 0, 0]
                
            return
        
        if not self.spi:
            return
        
        try:
            pattern = self.config["test_pattern"][self.current_pattern]
            
            # Define SRAM regions
            sram_size = 128 * 1024  # 128KB chip
            region_size = sram_size // 4
            
            total_bit_flips = 0
            max_consecutive = 0
            current_consecutive = 0
            region_errors = [0, 0, 0, 0]
            
            # Check each region for errors
            for region in range(4):
                start_addr = region * region_size
                end_addr = (region + 1) * region_size
                
                # Check sample of addresses in each region
                sample_size = min(1000, region_size)  # Check 1000 random addresses
                sample_addrs = np.random.choice(range(start_addr, end_addr), sample_size, replace=False)
                
                region_flips = 0
                
                for addr in sample_addrs:
                    # Read byte
                    data = self.read_sram(addr, 1)[0]
                    
                    # Check for bit flips
                    bit_flips = bin(data ^ pattern).count('1')
                    total_bit_flips += bit_flips
                    region_flips += bit_flips
                    
                    # Track consecutive errors
                    if bit_flips > 0:
                        current_consecutive += 1
                    else:
                        current_consecutive = 0
                    
                    max_consecutive = max(max_consecutive, current_consecutive)
                
                # Scale up region errors based on sample
                scale_factor = region_size / sample_size
                region_errors[region] = int(region_flips * scale_factor)
                
                # Reset pattern for next check
                if bit_flips > 0:
                    for addr in sample_addrs:
                        self.write_sram(addr, [pattern])
            
            # Update data
            self.data["bit_flips_count"] = total_bit_flips
            self.data["max_run_length"] = max_consecutive
            self.data["sram_regions"] = region_errors
            
            # Toggle pattern for next check
            self.current_pattern = (self.current_pattern + 1) % len(self.config["test_pattern"])
            
        except Exception as e:
            logger.error(f"SRAM error check failed: {str(e)}")
      def read_sensors(self):
        """Read all sensor data and update the data structure"""
        if self.config["simulation_mode"]:
            self._simulate_sensor_data()
            return
        
        # Update timestamp
        self.data["timestamp"] = datetime.now().isoformat()
        
        # Keep track of which sensors were successfully read
        success = {
            "bmp280": False,
            "gps": False,
            "cosmic": False,
            "battery": False
        }
        
        # Read BMP280 sensor (temperature, pressure, altitude)
        try:
            if self.bmp280:
                self.data["temperature"] = round(self.bmp280.temperature, 2)
                self.data["pressure"] = round(self.bmp280.pressure, 2)
                self.data["altitude"] = round(self.bmp280.altitude, 2)
                logger.debug(f"BMP280: Temp={self.data['temperature']:.2f}°C, "
                           f"Press={self.data['pressure']:.2f}hPa, Alt={self.data['altitude']:.1f}m")
                success["bmp280"] = True
        except Exception as e:
            logger.error(f"BMP280 sensor read error: {str(e)}")
            # Keep previous values
            self.error_count += 1
        
        # Read GPS data
        try:
            if self.gps_serial and self.gps_serial.is_open:
                valid_sentence = False
                for _ in range(10):  # Try up to 10 NMEA sentences
                    try:
                        line = self.gps_serial.readline().decode('ascii', errors='ignore').strip()
                        if not line:
                            continue
                            
                        # Handle GPGGA sentences (position and altitude)
                        if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
                            parts = line.split(',')
                            if len(parts) >= 10 and parts[2] and parts[4]:
                                try:
                                    lat = self._parse_gps_coordinate(parts[2], parts[3])
                                    lon = self._parse_gps_coordinate(parts[4], parts[5])
                                    
                                    if parts[9] and parts[9] != '':
                                        alt = float(parts[9])
                                        self.data["gps_altitude"] = alt
                                    
                                    self.data["latitude"] = lat
                                    self.data["longitude"] = lon
                                    logger.debug(f"GPS: Lat={lat:.6f}, Lon={lon:.6f}, Alt={self.data['gps_altitude']:.1f}m")
                                    valid_sentence = True
                                    break
                        
                        # Handle GPRMC sentences (time and date)
                        elif line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                            parts = line.split(',')
                            if len(parts) >= 10 and parts[2] == 'A':  # 'A' means data valid
                                valid_sentence = True
                    except Exception as e:
                        logger.debug(f"GPS parse error: {str(e)}")
                
                success["gps"] = valid_sentence
        except Exception as e:
            logger.error(f"GPS sensor read error: {str(e)}")
            self.error_count += 1
        
        # Read cosmic ray count
        try:
            self.data["cosmic_ray_count"] = self.cosmic_ray_count
            self.cosmic_ray_count = 0
            success["cosmic"] = True
        except Exception as e:
            logger.error(f"Cosmic ray counter error: {str(e)}")
            self.error_count += 1
        
        # Read battery voltage using ADC if available
        try:
            if HAS_HARDWARE:
                # This would use an ADC connected to battery through voltage divider
                # For now, we're using a simulated value or fixed value
                self.data["battery_voltage"] = 9.0  # Placeholder
                
                # If MCP3008 or similar ADC is connected, we would read it like this:
                # import Adafruit_MCP3008
                # adc = Adafruit_MCP3008.MCP3008(clk=11, cs=8, miso=9, mosi=10)
                # raw_value = adc.read_adc(0)
                # voltage = raw_value * (3.3 / 1023) * voltage_divider_factor
                # self.data["battery_voltage"] = voltage
            else:
                # Simulate battery drain over time
                uptime = time.time() - self.start_time
                hours_running = uptime / 3600
                # Start at 9V, drain to 7V over 48 hours
                self.data["battery_voltage"] = max(7.0, 9.0 - (hours_running / 48) * 2.0)
            
            success["battery"] = True
        except Exception as e:
            logger.error(f"Battery monitoring error: {str(e)}")
            self.error_count += 1
        
        # Log sensor status
        if all(success.values()):
            logger.debug("All sensors read successfully")
        else:
            failed = [name for name, status in success.items() if not status]
            if failed:
                logger.warning(f"Failed to read sensors: {', '.join(failed)}")
    
    def _parse_gps_coordinate(self, coord_str, direction):
        """Parse NMEA GPS coordinate format
        
        Args:
            coord_str: GPS coordinate string in DDMM.MMMM format
            direction: Direction indicator (N, S, E, W)
            
        Returns:
            Decimal degree format of the coordinate
        """
        if not coord_str:
            return 0.0
        
        # Format is DDMM.MMMM
        try:
            # Ensure we have valid string input
            coord_str = coord_str.strip()
            if not coord_str or '.' not in coord_str:
                return 0.0
                
            dot_position = coord_str.index('.')
            
            # Determine if this is latitude (2 digits for degrees) or longitude (3 digits)
            # For lat: DDMM.MMMM, For lon: DDDMM.MMMM
            degree_digits = 2 if dot_position <= 4 else 3
            
            degrees = int(coord_str[:degree_digits])
            minutes = float(coord_str[degree_digits:])
            
            # Convert to decimal degrees
            decimal = degrees + minutes / 60
            
            # Apply direction
            if direction in ['S', 'W']:
                decimal = -decimal
                
            return round(decimal, 6)
        except Exception as e:
            logger.error(f"GPS coordinate parse error: {str(e)}")
        
        return 0.0
    
    def _simulate_sensor_data(self):
        """Generate simulated sensor data for testing without hardware
        
        Simulates a realistic balloon flight profile with ascent, float, and descent phases.
        """
        # Calculate elapsed time - use time since script start for repeatable patterns
        sim_elapsed = time.time() - self.sim_start_time if hasattr(self, 'sim_start_time') else 0
        
        # Reset simulation every 3 hours
        sim_elapsed = sim_elapsed % (3 * 60 * 60)
        
        # Generate altitude profile (ascent, float, descent)
        max_altitude = 32000  # 32km (typical high altitude balloon)
        ascent_time = 90 * 60  # 90 minutes ascent
        float_time = 45 * 60   # 45 minutes float
        descent_time = 45 * 60  # 45 minutes descent
        total_time = ascent_time + float_time + descent_time
        
        # Define more realistic ascent rate curve
        # Initially faster (3-5 m/s) and slowing as altitude increases
        if sim_elapsed <= ascent_time:
            # Non-linear ascent (slowing as altitude increases)
            progress = sim_elapsed / ascent_time
            altitude = max_altitude * (1 - (1 - progress) ** 1.5)
            
            # Add realistic variation to ascent rate
            if hasattr(self, 'last_altitude') and hasattr(self, 'last_time'):
                time_delta = time.time() - self.last_time
                if time_delta > 0:
                    ascent_rate = (altitude - self.last_altitude) / time_delta
                    # Add small random variations to ascent rate
                    altitude += np.random.normal(0, ascent_rate * 0.1)
            
        # Float phase with small oscillations
        elif sim_elapsed <= (ascent_time + float_time):
            float_progress = (sim_elapsed - ascent_time) / float_time
            
            # Add realistic float variations (vertical oscillations)
            oscillation = 500 * np.sin(float_progress * np.pi * 2)
            slower_oscillation = 300 * np.sin(float_progress * np.pi * 0.5)
            altitude = max_altitude + oscillation + slower_oscillation
            
        # Descent phase (faster initial descent)
        else:
            descent_progress = (sim_elapsed - ascent_time - float_time) / descent_time
            # Exponential descent (initially faster due to thinner air)
            altitude = max_altitude * (1 - descent_progress ** 0.6)
        
        # Record for rate calculations
        self.last_altitude = altitude
        self.last_time = time.time()
        
        # Add realistic noise
        altitude += np.random.normal(0, 30)
        altitude = max(0, altitude)
        
        # Calculate temperature based on altitude using international standard atmosphere
        # Temperature decreases linearly in troposphere, constant in lower stratosphere
        sea_level_temp = 15 + 5 * np.sin(sim_elapsed / 3600 * np.pi)  # Add daily variation
        if altitude <= 11000:  # Troposphere
            temperature = sea_level_temp - 6.5 * (altitude / 1000)
        elif altitude <= 20000:  # Lower stratosphere
            temperature = -56.5  # Constant temperature in lower stratosphere
        else:  # Upper stratosphere
            temperature = -56.5 + 0.001 * (altitude - 20000)  # Slight increase
        
        # Add noise to temperature
        temperature += np.random.normal(0, 0.3)
        
        # Calculate pressure based on altitude
        # Barometric formula
        if altitude <= 11000:  # Troposphere
            pressure = 1013.25 * (1 - 0.0065 * altitude / 288.15) ** 5.255
        else:  # Stratosphere
            pressure = 226.32 * np.exp(-0.000157 * (altitude - 11000))
        
        # Add noise to pressure
        pressure += np.random.normal(0, pressure * 0.005)
        pressure = max(0.1, pressure)
        
        # Simulate GPS coordinates (spiral path)
        # Start at a random location and spiral outward
        start_lat, start_lon = 39.0, -98.0  # Approximate center of USA
        if not hasattr(self, 'sim_lat_offset'):
            self.sim_lat_offset = np.random.uniform(-0.05, 0.05)
            self.sim_lon_offset = np.random.uniform(-0.05, 0.05)
        
        # Spiral pattern
        radius = 0.05 * (sim_elapsed / total_time)
        angle = (sim_elapsed / 60) * 2 * np.pi  # Complete rotation every 1 minute
        
        lat = start_lat + self.sim_lat_offset + radius * np.cos(angle)
        lon = start_lon + self.sim_lon_offset + radius * np.sin(angle)
        
        # Add GPS noise (more noise at higher altitudes to simulate poor reception)
        gps_noise_factor = min(1.0, altitude / 20000)  # More noise above 20km
        lat += np.random.normal(0, 0.0005 * gps_noise_factor)
        lon += np.random.normal(0, 0.0005 * gps_noise_factor)
        
        # Simulate cosmic ray count (increases with altitude exponentially)
        # Cosmic ray flux roughly doubles every 1.5km above sea level
        sea_level_flux = 1.0
        cosmic_ray_base = sea_level_flux * np.exp(altitude / 5000)
        
        # Add randomness following Poisson distribution (for discrete events)
        cosmic_ray_count = np.random.poisson(cosmic_ray_base)
        
        # Add occasional cosmic ray shower events
        if np.random.random() < 0.01:  # 1% chance per measurement
            cosmic_ray_count *= np.random.randint(2, 10)
            logger.debug("Simulated cosmic ray shower event")
        
        # Calculate realistic battery drain
        # Factors: temperature (cold reduces voltage), time
        battery_nominal = 9.0  # Starting voltage
        time_factor = sim_elapsed / (48 * 3600)  # 48 hour expected life
        temp_factor = 0.0
        if temperature < -30:
            temp_factor = (temperature + 30) * 0.01  # 1% reduction per degree below -30C
            
        self.data["battery_voltage"] = max(6.5, battery_nominal * (1.0 - time_factor - temp_factor))
        
        # Update data in the main data structure
        self.data["timestamp"] = datetime.now().isoformat()
        self.data["altitude"] = round(altitude, 2)
        self.data["temperature"] = round(temperature, 2)
        self.data["pressure"] = round(pressure, 2)
        self.data["latitude"] = round(lat, 6)
        self.data["longitude"] = round(lon, 6)
        self.data["gps_altitude"] = round(altitude + np.random.normal(0, 100), 2)  # GPS altitude is less accurate
        self.data["cosmic_ray_count"] = cosmic_ray_count
    
    def log_data(self):
        """Log current data"""
        # Calculate cosmic intensity based on altitude
        # (simple exponential model)
        altitude = self.data["altitude"]
        cosmic_intensity = np.exp(altitude / 10000)
        
        # Create data packet with cosmic intensity added
        data_packet = self.data.copy()
        data_packet["cosmic_intensity"] = cosmic_intensity
        
        # Add to queue for processing
        data_queue.put(data_packet)
        
        # Log to console
        logger.info(f"Alt:{data_packet['altitude']:.1f}m, Temp:{data_packet['temperature']:.1f}C, SEUs:{data_packet['bit_flips_count']}")
      def get_system_health(self) -> Dict:
        """Get system health information
        
        Returns:
            Dict containing system health metrics
        """
        uptime = time.time() - self.start_time
        self.data["uptime"] = int(uptime)
        
        # Calculate memory usage
        memory_info = {}
        try:
            import psutil
            process = psutil.Process()
            memory_info = {
                "rss_mb": process.memory_info().rss / (1024 * 1024),
                "vms_mb": process.memory_info().vms / (1024 * 1024),
                "percent": process.memory_percent()
            }
        except ImportError:
            # psutil not available, use minimal info
            memory_info = {"available": False}
        
        # Get CPU temperature (Raspberry Pi specific)
        cpu_temp = None
        if HAS_HARDWARE:
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    cpu_temp = float(f.read().strip()) / 1000
            except:
                pass
        
        # System health data
        health = {
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "memory": memory_info,
            "cpu_temperature": cpu_temp,
            "error_count": self.error_count,
            "sram_checks": self.last_sram_check - self.start_time,
            "data_points": len(self.sim_data_points) if hasattr(self, 'sim_data_points') else 0,
            "config": {k: v for k, v in self.config.items() if k not in ["api_key", "mqtt_password"]}
        }
        
        return health
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime seconds into a human-readable string
        
        Args:
            seconds: Uptime in seconds
            
        Returns:
            Formatted uptime string
        """
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{int(days)}d")
        if hours > 0 or days > 0:
            parts.append(f"{int(hours)}h")
        if minutes > 0 or hours > 0 or days > 0:
            parts.append(f"{int(minutes)}m")
        
        parts.append(f"{int(seconds)}s")
        return " ".join(parts)
    
    def _update_simulation_visualization(self):
        """Update the simulation visualization with current data points"""
        if not hasattr(self, 'sim_figure') or self.sim_figure is None:
            return
            
        try:
            # Extract data for plotting
            times = [t - self.sim_start_time for t, _, _, _, _ in self.sim_data_points]
            altitudes = [a for _, a, _, _, _ in self.sim_data_points]
            temperatures = [t for _, _, t, _, _ in self.sim_data_points]
            seus = [s for _, _, _, s, _ in self.sim_data_points]
            cosmic = [c for _, _, _, _, c in self.sim_data_points]
            
            # Update plot data
            self.sim_altitude_line.set_data(times, altitudes)
            self.sim_temp_line.set_data(times, temperatures)
            self.sim_seu_line.set_data(times, seus)
            self.sim_cosmic_line.set_data(times, cosmic)
            
            # Adjust axis limits
            for ax in self.sim_figure.axes:
                ax.relim()
                ax.autoscale_view()
                
            # Update the plot
            self.sim_figure.canvas.draw_idle()
            self.sim_figure.canvas.flush_events()
        except Exception as e:
            logger.debug(f"Visualization update error: {str(e)}")
    
    def run(self):
        """Main run loop"""
        logger.info("Starting SEU detector")
        health_check_interval = 60  # Check system health every minute
        last_health_check = time.time()
        
        try:
            while running:
                current_time = time.time()
                
                # Check if it's time to read sensors
                if current_time - self.last_sensor_read >= self.config["sensor_read_interval"]:
                    self.read_sensors()
                    self.last_sensor_read = current_time
                
                # Check if it's time to check SRAM
                if current_time - self.last_sram_check >= self.config["sram_check_interval"]:
                    self.check_sram_errors()
                    self.last_sram_check = current_time
                
                # Check if it's time to log data
                if current_time - self.last_log_time >= self.config["log_interval"]:
                    self.log_data()
                    self.last_log_time = current_time
                    
                    # Store data point for simulation visualization
                    if self.config["simulation_mode"] and hasattr(self, 'sim_data_points'):
                        self.sim_data_points.append((
                            current_time,
                            self.data["altitude"],
                            self.data["temperature"],
                            self.data["bit_flips_count"],
                            self.data["cosmic_ray_count"]
                        ))
                        
                        # Keep a reasonable history
                        if len(self.sim_data_points) > 1000:
                            self.sim_data_points = self.sim_data_points[-1000:]
                        
                        # Update visualization if enabled
                        if self.config.get("visualize_simulation", False):
                            self._update_simulation_visualization()
                
                # Periodic system health check
                if current_time - last_health_check >= health_check_interval:
                    health = self.get_system_health()
                    logger.debug(f"System health: Uptime={health['uptime_formatted']}, "
                                f"Errors={health['error_count']}")
                    last_health_check = current_time
                
                # Sleep a bit to avoid CPU hogging
                # Use shorter sleep in battery saving mode
                if self.config.get("battery_saving", False):
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in run loop: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Auto-restart if enabled and not too many errors
            if self.config.get("restart_on_error", True) and self.error_count < 10:
                logger.info("Attempting to continue after error...")
                return self.run()
            else:
                raise
    
    def cleanup(self):
        """Clean up hardware resources"""
        logger.info("Cleaning up resources...")
        
        if not self.config["simulation_mode"]:
            # Close hardware interfaces
            if self.spi:
                try:
                    self.spi.close()
                    logger.debug("SPI interface closed")
                except:
                    pass
            
            if self.gps_serial:
                try:
                    self.gps_serial.close()
                    logger.debug("GPS serial closed")
                except:
                    pass
            
            # Clean up GPIO
            try:
                GPIO.cleanup()
                logger.debug("GPIO cleaned up")
            except:
                pass
        
        # Close visualization if active
        if hasattr(self, 'sim_figure') and self.sim_figure:
            try:
                import matplotlib.pyplot as plt
                plt.close(self.sim_figure)
                logger.debug("Simulation visualization closed")
            except:
                pass
        
        logger.info("SEU detector cleaned up successfully")

def data_processing_thread(config):
    """Thread for processing data from queue and saving/publishing"""
    logger.info("Starting data processing thread")
    
    # Shared variables for thread
    daily_file = None
    current_day = None
    mqtt_client = None
    pending_files = []
    last_api_send = time.time()
    data_buffer = []  # Buffer for batching API requests
    
    # Set up MQTT if enabled
    if config["mqtt_enabled"]:
        try:
            mqtt_client = mqtt.Client()
            # Add authentication if provided
            if "mqtt_username" in config and "mqtt_password" in config:
                mqtt_client.username_pw_set(config["mqtt_username"], config["mqtt_password"])
            
            # Add TLS if enabled
            if config.get("mqtt_use_tls", False):
                mqtt_client.tls_set()
            
            mqtt_client.connect(config["mqtt_broker"], config["mqtt_port"], keepalive=60)
            mqtt_client.loop_start()
            logger.info(f"Connected to MQTT broker at {config['mqtt_broker']}:{config['mqtt_port']}")
        except Exception as e:
            logger.error(f"MQTT connection error: {str(e)}")
            mqtt_client = None
    
    # Function to compress old data files
    def compress_old_files():
        """Compress data files older than 24 hours"""
        if not config.get("compress_old_files", False):
            return
        
        try:
            import gzip
            import shutil
            from datetime import timedelta
            
            # Get list of CSV files in data directory
            data_path = config["data_storage_path"]
            now = datetime.now()
            
            for filename in os.listdir(data_path):
                if not filename.endswith(".csv") or filename.endswith(".gz"):
                    continue
                
                file_path = os.path.join(data_path, filename)
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Check if file is older than 24 hours
                if now - file_mod_time > timedelta(hours=24):
                    gz_path = file_path + ".gz"
                    logger.debug(f"Compressing old file: {filename}")
                    
                    # Compress file
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(gz_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # Check if compression worked
                    if os.path.exists(gz_path) and os.path.getsize(gz_path) > 0:
                        os.remove(file_path)
                        logger.info(f"Compressed old file: {filename}")
                    else:
                        logger.error(f"Failed to compress {filename}")
        except Exception as e:
            logger.error(f"Error compressing files: {str(e)}")
    
    # Function to send data to remote API
    def send_to_api(data_batch):
        """Send data batch to remote API"""
        if not config.get("remote_api_enabled", False) or not HAS_REQUESTS:
            return False
        
        api_url = config.get("remote_api_url", "")
        if not api_url:
            return False
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Device-ID": data_batch[0].get("device_id", "unknown")
            }
            
            # Add authentication if provided
            if "api_key" in config:
                headers["Authorization"] = f"Bearer {config['api_key']}"
            
            response = requests.post(api_url, json=data_batch, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.debug(f"Sent {len(data_batch)} records to API successfully")
                return True
            else:
                logger.warning(f"API returned error: {response.status_code} - {response.text}")
                return False
        except requests.exceptions.ConnectionError:
            logger.debug("API connection error (will retry later)")
            return False
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return False
    
    # Try to perform initial compression of old files
    compress_old_files()
    
    while running:
        try:
            # Get data from queue with timeout
            try:
                data = data_queue.get(timeout=1.0)
                data_queue.task_done()
            except queue.Empty:
                # Periodically check for files to compress
                if config.get("compress_old_files", False):
                    if random.random() < 0.01:  # ~1% chance to check on each iteration
                        compress_old_files()
                
                # Send buffered data to API if enough time has passed
                if (config.get("remote_api_enabled", False) and 
                    time.time() - last_api_send > 60 and  # Send at most once per minute
                    data_buffer):
                    if send_to_api(data_buffer):
                        data_buffer = []  # Clear buffer if successful
                        last_api_send = time.time()
                
                continue
            
            # Check if we need a new data file (daily files)
            day_str = datetime.now().strftime("%Y-%m-%d")
            if config.get("save_daily_files", True) and day_str != current_day:
                current_day = day_str
                if daily_file:
                    daily_file.close()
                    logger.debug("Closed previous data file")
                
                filename = os.path.join(config["data_storage_path"], f"seu_data_{day_str}.csv")
                new_file = not os.path.exists(filename)
                
                try:
                    daily_file = open(filename, 'a')
                    
                    # Write header if new file
                    if new_file:
                        header = "timestamp,altitude,temperature,pressure,latitude,longitude,"
                        header += "gps_altitude,bit_flips_count,max_run_length,cosmic_ray_count,"
                        header += "battery_voltage,cosmic_intensity,sram_region_0,sram_region_1,"
                        header += "sram_region_2,sram_region_3,device_id,uptime,version\n"
                        daily_file.write(header)
                    
                    logger.info(f"Opened data file: {filename}")
                except Exception as e:
                    logger.error(f"Failed to open data file: {str(e)}")
                    daily_file = None
            
            # Format data for CSV
            if daily_file:
                try:
                    csv_line = (f"{data['timestamp']},{data['altitude']:.2f},{data['temperature']:.2f},"
                              f"{data['pressure']:.2f},{data['latitude']:.6f},{data['longitude']:.6f},"
                              f"{data['gps_altitude']:.2f},{data['bit_flips_count']},{data['max_run_length']},"
                              f"{data['cosmic_ray_count']},{data['battery_voltage']:.2f},{data.get('cosmic_intensity', 0):.4f},"
                              f"{data['sram_regions'][0]},{data['sram_regions'][1]},{data['sram_regions'][2]},"
                              f"{data['sram_regions'][3]},{data.get('device_id', 'unknown')},{data.get('uptime', 0)},"
                              f"{data.get('version', '1.0.0')}\n")
                    
                    # Write to CSV file
                    daily_file.write(csv_line)
                    daily_file.flush()
                except Exception as e:
                    logger.error(f"Failed to write to data file: {str(e)}")
            
            # Publish to MQTT if enabled
            if mqtt_client and config["mqtt_enabled"]:
                try:
                    mqtt_client.publish(config["mqtt_topic"], json.dumps(data))
                except Exception as e:
                    logger.error(f"MQTT publish error: {str(e)}")
            
            # Buffer for API if enabled
            if config.get("remote_api_enabled", False):
                data_buffer.append(data)
                
                # Send immediately if buffer is large enough
                if len(data_buffer) >= config.get("api_batch_size", 10):
                    if send_to_api(data_buffer):
                        data_buffer = []
                        last_api_send = time.time()
            
            # Output to serial for external systems
            if config.get("serial_output", True):
                try:
                    sys.stdout.write(json.dumps(data) + "\n")
                    sys.stdout.flush()
                except Exception as e:
                    logger.debug(f"Serial output error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Data processing error: {str(e)}")
            logger.debug(traceback.format_exc())
    
    # Cleanup
    logger.info("Cleaning up data processing thread")
    
    # Save any remaining buffered data
    if config.get("remote_api_enabled", False) and data_buffer:
        send_to_api(data_buffer)
    
    # Close data file
    if daily_file:
        try:
            daily_file.close()
            logger.debug("Closed data file")
        except Exception:
            pass
    
    # Disconnect from MQTT
    if mqtt_client and config["mqtt_enabled"]:
        try:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            logger.debug("Disconnected from MQTT broker")
        except Exception:
            pass
            
    logger.info("Data processing thread stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C and other signals gracefully"""
    global running
    logger.info(f"Signal {sig} received, stopping SEU detector...")
    running = False

def load_config(config_path=None):
    """Load configuration from file and merge with defaults
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Dict containing merged configuration
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from config path
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
            logger.info(f"Loaded configuration from {config_path}")
        except json.JSONDecodeError:
            logger.error(f"Error parsing configuration file: {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    return config

def main():
    """Main entry point"""
    global running
    running = True
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SEU Detector for Raspberry Pi")
    parser.add_argument("-c", "--config", help="Path to configuration file")
    parser.add_argument("-s", "--simulate", action="store_true", help="Run in simulation mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-o", "--output", help="Path for data output")
    parser.add_argument("--save-config", help="Save current config to file and exit")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    args = parser.parse_args()
    
    # Show version if requested
    if args.version:
        print(f"SEU Detector v{VERSION}")
        return 0
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger("SEU-Detector").setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.simulate:
        config["simulation_mode"] = True
    
    if args.output:
        config["data_storage_path"] = args.output
    
    # Save config if requested
    if args.save_config:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_config)), exist_ok=True)
            with open(args.save_config, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {args.save_config}")
            return 0
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return 1
    
    logger.info("Starting SEU detector system")
    
    # Start data processing thread
    try:
        data_thread = threading.Thread(target=data_processing_thread, args=(config,))
        data_thread.daemon = True
        data_thread.start()
        
        # Create and run SEU detector
        detector = SEUDetector(config)
        try:
            detector.run()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in SEU detector: {str(e)}")
            logger.debug(traceback.format_exc())
            return 1
        finally:
            detector.cleanup()
        
        # Wait for data thread to finish processing queue
        logger.info("Waiting for data processing to complete...")
        if data_thread.is_alive():
            data_thread.join(timeout=5.0)
        
        logger.info("SEU detector stopped")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to start system: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1

if __name__ == "__main__":
    main()
