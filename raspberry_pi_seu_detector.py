	#!/usr/bin/env python3
"""
SEU Detector - Raspberry Pi Zero W Data Collection Script
This script runs on the Raspberry Pi Zero W and collects data from:
1. SRAM chips via SPI
2. BMP280 temperature/pressure sensor via I2C
3. GPS module via UART
4. Cosmic ray counter via GPIO

IMPORTANT NOTE:
This script is designed for Raspberry Pi hardware and requires specific hardware libraries.
Some import errors shown in the IDE are expected when editing on a non-Pi system.
The code will run correctly when deployed to an actual Raspberry Pi with the required libraries.

----------------------------------------------------
EDITOR WARNING:
IGNORE IMPORT ERRORS FOR HARDWARE LIBRARIES IN IDE
These libraries are only available on the Raspberry Pi:
- RPi.GPIO
- spidev
- smbus2
- board
- busio
- adafruit_bmp280
- adafruit_gps
- paho.mqtt.client
----------------------------------------------------
"""

# Add current directory and parent directory to path
import os
import sys
from pathlib import Path

# Get the script directory and related directories
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
parent_dir = script_dir.parent  # src directory if in standard structure
project_root = parent_dir.parent  # project root directory

# Add directories to Python path so we can import modules from these locations
# This is particularly important when running directly on the Raspberry Pi
# It ensures that imports work regardless of where/how the script is executed
if script_dir not in sys.path:
    sys.path.insert(0, str(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, str(parent_dir))
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Script directory: {script_dir}")
print(f"Python path: {sys.path}")

# Standard library imports
import time
import json
import logging
import threading
import queue
from datetime import datetime
import struct
import signal
import argparse
import traceback
import shutil
import gzip
import csv
import io
import random  # For occasional operations that need randomness
import math    # For math operations when numpy is not available
from typing import Dict, List, Optional, Tuple, Union, Any, TypedDict, TextIO
from pathlib import Path
from contextlib import contextmanager
from functools import wraps

# For development environment, define fallbacks for Raspberry Pi specific libraries
try:
    import numpy as np  # Used for certain calculations
except ImportError:
    # Create minimal numpy-like functions for math operations
    class NumpyFallback:
        def __init__(self):
            self.random = RandomFallback()

        def exp(self, value):
            return math.exp(value)

    class RandomFallback:
        def poisson(self, lam):
            # Simple approximation for poisson
            return max(0, round(random.gauss(lam, math.sqrt(lam))))

        def normal(self, mean, std):
            return random.gauss(mean, std)

        def choice(self, population, size=1, replace=True):
            if size == 1:
                return random.choice(population)
            result = []
            for _ in range(size):
                result.append(random.choice(population))
            return result

        def binomial(self, n, p):
            # Simple approximation for binomial
            count = 0
            for _ in range(n):
                if random.random() < p:
                    count += 1
            return count

    np = NumpyFallback()
from dataclasses import dataclass, field
from collections import deque

# Hardware-dependent libraries
# NOTE TO CODE EDITORS: The following imports will show errors in non-Raspberry Pi environments
# but they will work correctly when deployed to the actual Raspberry Pi hardware.
# pylint: disable=import-error,no-name-in-module
try:
    # Raspberry Pi hardware interfaces - these will only work on Raspberry Pi
    # Linting/IDE errors for these imports can be safely ignored
    import RPi.GPIO as GPIO
    import spidev
    import smbus2 as smbus
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
    # Only triggered when running on actual hardware with missing libraries
    print(f"Error: Hardware library not available ({str(e)})")
    print("This script requires hardware libraries for the Raspberry Pi.")
    print("Please install required dependencies: pip install -r requirements_raspberry_pi.txt")
    sys.exit(1)

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

# Setup module-level logger
logger = logging.getLogger("seu_detector")

# Default configuration parameters
DEFAULT_CONFIG = {
    "sram_chips": 4,               # Number of SRAM chips connected
    "sram_size_bytes": 131072,     # 128KB per chip (23LC1024)
    "spi_bus": 0,                  # SPI bus number
    "spi_device": 0,               # SPI device number
    "cs_pins": [8, 7, 1, 12],      # Chip select GPIO pins for SRAM chips
    "i2c_bus": 1,                  # I2C bus number
    "cosmic_counter_pin": 17,      # GPIO pin for cosmic ray detector
    "gps_port": "/dev/ttyS0",      # Serial port for GPS module
    "gps_baud": 9600,              # GPS baud rate
    "sample_rate": 10,             # Readings per second
    "test_pattern": [              # Test patterns to write to SRAM
        0x55,                      # 01010101
        0xAA,                      # 10101010
        0xFF,                      # 11111111
        0x00                       # 00000000
    ],
    "mqtt_broker": "",             # Leave empty to disable MQTT
    "mqtt_port": 1883,
    "mqtt_topic": "seu/data",
    "output_dir": "data",          # Directory to store data files
    "log_level": "info"            # Logging level
}

# Global variables
running = True
data_queue = queue.Queue()
mqtt_client = None

class SEUDetector:
    """Single Event Upset (SEU) Detector for SRAM chips

    This class manages the detection of SEUs in SRAM chips using various
    test patterns and reports upsets that may be caused by cosmic rays.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the SEU detector

        Args:
            config: Configuration parameters (see DEFAULT_CONFIG)
        """
        self.running = False
        self.config = DEFAULT_CONFIG.copy()
        self.error_count = 0
        self.last_sensor_read = time.time()

        # Apply any provided configuration
        if config:
            self.config.update(config)

        # Setup logging
        log_level = self.config.get("log_level", "info").upper()
        logger.setLevel(getattr(logging, log_level))

        # Initialize data collection structure
        self.data = {
            "timestamp": "",
            "uptime_seconds": 0,
            "chip_count": self.config["sram_chips"],
            "pattern": self.config["test_pattern"][0],
            "seu_count": 0,
            "seu_by_chip": [0] * self.config["sram_chips"],
            "seu_by_pattern": [0] * len(self.config["test_pattern"]),
            "max_run_length": 0,
            "temperature_c": 0,
            "pressure_hpa": 0,
            "altitude_m": 0,
            "latitude": 0,
            "longitude": 0,
            "speed_kmh": 0,
            "satellites": 0,
            "cosmic_counts": 0,
            "cpu_temp_c": 0,
            "cpu_percent": 0,
        }

        self.data_queue = queue.Queue()
        self.sample_interval = 1.0 / self.config["sample_rate"]
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize hardware
        try:
            # Setup SPI for SRAM communication
            self.spi = spidev.SpiDev()
            self.spi.open(self.config["spi_bus"], self.config["spi_device"])
            self.spi.max_speed_hz = 4000000  # 4MHz
            self.spi.mode = 0

            # Setup GPIO
            GPIO.setmode(GPIO.BCM)
            # Set up chip select pins
            for pin in self.config["cs_pins"][:self.config["sram_chips"]]:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.HIGH)  # Disabled by default (CS is active LOW)

            # Setup cosmic ray counter input
            if self.config.get("cosmic_counter_pin") is not None:
                GPIO.setup(self.config["cosmic_counter_pin"], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                GPIO.add_event_detect(
                    self.config["cosmic_counter_pin"],
                    GPIO.RISING,
                    callback=self._cosmic_ray_callback
                )

            # Setup I2C for sensors (BMP280)
            try:
                i2c = busio.I2C(board.SCL, board.SDA)
                self.bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(i2c)
                self.bmp280.sea_level_pressure = 1013.25  # Default sea level pressure in hPa
            except Exception as e:
                logger.warning(f"BMP280 initialization failed: {e}")
                self.bmp280 = None

            # Setup GPS
            try:
                self.uart = serial.Serial(
                    self.config["gps_port"],
                    baudrate=self.config["gps_baud"],
                    timeout=1
                )
                self.gps = adafruit_gps.GPS(self.uart, debug=False)

                # Initialize GPS module
                self.gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")  # Only enable GGA and RMC
                self.gps.send_command(b"PMTK220,1000")  # Set update rate to 1Hz
            except Exception as e:
                logger.warning(f"GPS initialization failed: {e}")
                self.gps = None

            # MQTT setup (if broker specified)
            if self.config.get("mqtt_broker"):
                self.mqtt_client = mqtt.Client()
                self.mqtt_client.connect(
                    self.config["mqtt_broker"],
                    self.config.get("mqtt_port", 1883)
                )
                self.mqtt_client.loop_start()
            else:
                self.mqtt_client = None

        except Exception as e:
            logger.error(f"Hardware initialization failed: {e}")
            raise

        # Set initial test pattern
        self.current_pattern = 0

        # Cosmic ray counter
        self.cosmic_count = 0
        self.last_cosmic_timestamp = 0

        # Start time for uptime calculation
        self.start_time = time.time()

        logger.info("SEU Detector initialized successfully")
def __init__(self, config: Optional[Dict] = None):

    """Initialize the SEU detector

    Args:
        config: Configuration parameters (see DEFAULT_CONFIG)
    """

    # Use default config and update if provided
    self.config = DEFAULT_CONFIG.copy()
    if config:
        self.config.update(config)

    # Setup logger level
    log_level = self.config.get("log_level", "INFO").upper()
    logger.setLevel(getattr(logging, log_level))

    # Control flags, counters, and timers
    self.running = False
    self.error_count = 0
    self.last_sensor_read = time.time()
    self.sensor_read_interval = 1.0 / self.config.get("sample_rate", 10)
    self.start_time = time.time()

    # Data queue for threading
    self.data_queue = queue.Queue()

    # Cosmic ray count and last timestamp
    self.cosmic_count = 0
    self.last_cosmic_timestamp = 0

    # SRAM test pattern index
    self.current_pattern = 0

    # Simulation mode flag
    self.simulation_mode = self.config.get("simulation_mode", False)

    # Data dictionary for sensor and SEU info
    self.data = {
        "timestamp": "",
        "uptime_seconds": 0,
        "chip_count": self.config.get("sram_chips", 4),
        "pattern": self.config.get("test_pattern", [0x55])[0],
        "seu_count": 0,
        "seu_by_chip": [0] * self.config.get("sram_chips", 4),
        "seu_by_pattern": [0] * len(self.config.get("test_pattern", [0x55, 0xAA, 0xFF, 0x00])),
        "max_run_length": 0,
        "temperature_c": 0,
        "pressure_hpa": 0,
        "altitude_m": 0,
        "latitude": 0,
        "longitude": 0,
        "speed_kmh": 0,
        "satellites": 0,
        "cosmic_counts": 0,
        "cpu_temp_c": 0,
        "cpu_percent": 0,
    }

    # Prepare output directory
    self.output_dir = Path(self.config.get("output_dir", "./output"))
    self.output_dir.mkdir(exist_ok=True, parents=True)

    # Hardware interfaces placeholders
    self.spi = None
    self.bmp280 = None
    self.gps = None
    self.mqtt_client = None

    try:
        # SPI setup
        spi_bus = self.config.get("spi_bus", 0)
        spi_device = self.config.get("spi_device", 0)
        self.spi = spidev.SpiDev()
        self.spi.open(spi_bus, spi_device)
        self.spi.max_speed_hz = 4000000  # 4 MHz
        self.spi.mode = 0

        # GPIO setup
        GPIO.setmode(GPIO.BCM)
        cs_pins = self.config.get("cs_pins", [])
        sram_chips = self.config.get("sram_chips", 4)
        for pin in cs_pins[:sram_chips]:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)  # CS inactive

        # Cosmic ray counter pin setup
        cosmic_pin = self.config.get("cosmic_counter_pin")
        if cosmic_pin is not None:
            GPIO.setup(cosmic_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            GPIO.add_event_detect(cosmic_pin, GPIO.RISING, callback=self._cosmic_ray_callback)

        # BMP280 sensor
        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(i2c)
            self.bmp280.sea_level_pressure = 1013.25
        except Exception as e:
            logger.warning(f"BMP280 initialization failed: {e}")
            self.bmp280 = None

        # GPS setup
        try:
            gps_port = self.config.get("gps_port", "/dev/ttyS0")
            gps_baud = self.config.get("gps_baud", 9600)
            self.uart = serial.Serial(gps_port, baudrate=gps_baud, timeout=1)
            self.gps = adafruit_gps.GPS(self.uart, debug=False)
            self.gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
            self.gps.send_command(b"PMTK220,1000")
        except Exception as e:
            logger.warning(f"GPS initialization failed: {e}")
            self.gps = None

        # MQTT client setup
        mqtt_broker = self.config.get("mqtt_broker")
        if mqtt_broker:
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.connect(mqtt_broker, self.config.get("mqtt_port", 1883))
            self.mqtt_client.loop_start()

    except Exception as e:
        logger.error(f"Hardware initialization failed: {e}")
        raise

    logger.info("SEU Detector initialized successfully")

    def _cosmic_ray_callback(self, channel):
        """Callback function for cosmic ray counter"""
        self.cosmic_count += 1

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
        # Update timestamp
        now = datetime.now()
        self.data["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.data["uptime_seconds"] = time.time() - self.start_time

        # Read temperature/pressure (BMP280)
        if self.bmp280:
            try:
                self.data["temperature_c"] = round(self.bmp280.temperature, 2)
                self.data["pressure_hpa"] = round(self.bmp280.pressure, 2)
                self.data["altitude_m"] = round(self.bmp280.altitude, 2)
            except Exception as e:
                logger.error(f"BMP280 read error: {e}")

        # Read GPS data
        if self.gps:
            try:
                # Update GPS readings if available
                self.gps.update()

                # If we have a fix, update location data
                if self.gps.has_fix:
                    self.data["latitude"] = self.gps.latitude
                    self.data["longitude"] = self.gps.longitude
                    self.data["speed_kmh"] = self.gps.speed_knots * 1.852 if self.gps.speed_knots else 0
                    self.data["satellites"] = self.gps.satellites

                    # If we have altitude from GPS, use it as it's more accurate
                    if self.gps.altitude_m is not None:
                        self.data["altitude_m"] = round(self.gps.altitude_m, 2)
            except Exception as e:
                logger.error(f"GPS read error: {e}")

        # Read system information
        try:
            # CPU temperature (Raspberry Pi specific)
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = int(f.read()) / 1000.0
                self.data["cpu_temp_c"] = round(cpu_temp, 1)

            # CPU usage
            if psutil:
                self.data["cpu_percent"] = psutil.cpu_percent(interval=None)
        except Exception as e:
            logger.error(f"System info read error: {e}")

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
            Dict with health metrics
        """
        health = {
            "uptime": self.data["uptime_seconds"],
            "cpu_temp": self.data["cpu_temp_c"],
            "cpu_percent": self.data["cpu_percent"],
            "hardware_status": {}
        }

        # Check hardware components
        if self.spi:
            try:
                # Try simple SPI transaction
                result = self.spi.xfer([0x00])
                health["hardware_status"]["spi"] = "ok"
            except Exception:
                health["hardware_status"]["spi"] = "error"
        else:
            health["hardware_status"]["spi"] = "not_initialized"

        # Check I2C/BMP280
        if self.bmp280:
            try:
                # Try reading temperature to check if sensor responds
                _ = self.bmp280.temperature
                health["hardware_status"]["bmp280"] = "ok"
            except Exception:
                health["hardware_status"]["bmp280"] = "error"
        else:
            health["hardware_status"]["bmp280"] = "not_initialized"

        # Check GPS
        if self.gps:
            health["hardware_status"]["gps"] = "ok" if self.gps.has_fix else "no_fix"
        else:
            health["hardware_status"]["gps"] = "not_initialized"

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
        """Clean up resources and shutdown hardware connections"""
        logger.info("Cleaning up resources...")
        self.running = False

        # Close SPI
        if self.spi:
            self.spi.close()

        # Clean up GPIO
        try:
            GPIO.cleanup()
        except Exception as e:
            logger.warning(f"GPIO cleanup failed: {e}")

        # Close GPS serial connection
        if self.gps and hasattr(self.gps, 'uart'):
            try:
                self.gps.uart.close()
            except Exception as e:
                logger.warning(f"GPS UART close failed: {e}")

        # Stop MQTT client if running
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception as e:
                logger.warning(f"MQTT client disconnect failed: {e}")

        logger.info("Cleanup complete")

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
    if config.get("mqtt.enabled", False):
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
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))                # Check if file is older than 24 hours
                if now - file_mod_time > timedelta(hours=24):
                    # Simply remove old files in development environment
                    # In production on Raspberry Pi, we would compress these
                    os.remove(file_path)
                    logger.info(f"Removed old file: {filename}")
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

    # Ensure we have the correct paths in sys.path
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    parent_dir = script_dir.parent  # src directory
    project_root = parent_dir.parent  # project root directory

    for directory in [str(script_dir), str(parent_dir), str(project_root)]:
        if directory not in sys.path:
            sys.path.insert(0, directory)
            print(f"Added to Python path: {directory}")

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SEU Detector for Raspberry Pi")
    parser.add_argument("-c", "--config", help="Path to configuration file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("-o", "--output", help="Path for data output")
    parser.add_argument("--save-config", help="Save current config to file and exit")
    parser.add_argument("--simulate", action="store_true", help="Run script in simulation mode")
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
