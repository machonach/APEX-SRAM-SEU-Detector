"""
SEU Detector - Raspberry Pi Pico W Data Collection Script
This script runs on the Raspberry Pi Pico W and collects data from:
1. SRAM chips via SPI
2. BMP280 temperature/pressure sensor via I2C
3. GPS module via UART
4. Cosmic ray counter via GPIO

It then sends this data to the Raspberry Pi Zero 2 W via WiFi using MQTT or HTTP.

Required MicroPython modules:
- machine: For hardware access (built-in)
- network: For WiFi (built-in)
- time: For timing (built-in)
- umqtt.simple: For MQTT communication (install via upip)
- urequests: For HTTP requests (install via upip)
- bmp280: For BMP280 sensor (custom or adafruit)
"""

import machine
import network
import time
import json
import gc
import struct
import binascii
from machine import Pin, SPI, I2C, UART, Timer, ADC

# Try to import optional modules
try:
    import urequests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False
    print("urequests module not found - HTTP transmission disabled")

try:
    from umqtt.simple import MQTTClient
    HAVE_MQTT = True
except ImportError:
    HAVE_MQTT = False
    print("umqtt.simple module not found - MQTT transmission disabled")

# Constants
VERSION = "1.0.0"
DEFAULT_SAMPLE_RATE = 10  # Hz
DEFAULT_LOG_INTERVAL = 5  # seconds
DEVICE_ID = "pico-seu-1"  # Unique ID for this Pico

# Default configuration
CONFIG = {
    "sram_chips": 4,
    "sram_size_bytes": 131072,  # 128KB per chip (23LC1024)
    "cs_pins": [5, 6, 7, 8],    # Chip select GPIO pins for SRAM chips
    "cosmic_counter_pin": 22,   # GPIO pin for cosmic ray detector
    "test_pattern": [           # Test patterns to write to SRAM
        0x55,                   # 01010101
        0xAA,                   # 10101010
        0xFF,                   # 11111111
        0x00                    # 00000000
    ],
    "sample_rate": DEFAULT_SAMPLE_RATE,    # Readings per second
    "log_interval": DEFAULT_LOG_INTERVAL,  # Seconds between data transmissions
    "wifi_ssid": "",           # WiFi SSID - set before running
    "wifi_password": "",       # WiFi Password - set before running
    "use_mqtt": True,          # Use MQTT for data transmission
    "mqtt_broker": "192.168.1.XXX",  # Address of Pi Zero 2 W
    "mqtt_port": 1883,
    "mqtt_topic": "seu/pico/data",
    "http_server": "http://192.168.1.XXX:8000/api/data",  # REST API on Pi Zero
    "simulation_mode": False,  # If true, generate synthetic data
    "led_pin": 25,            # Onboard LED pin for status
    "sample_rate": 1,              # Reduce to 1Hz to save power
    "log_interval": 60,            # Log data every minute
    "use_mqtt": False,             # Disable MQTT in flight (no WiFi at altitude)
    "offline_mode": True,          # Enable offline data storage
    "high_altitude_mode": True,    # Enable high altitude optimizations
    "power_saving": True,          # Enable power saving features
    "max_file_size_mb": 10,        # Split logs into manageable chunks
    "battery_monitor_pin": 28,     # ADC pin for battery monitoring
    "gps_power_save": True,        # Enable GPS power saving when appropriate
}


# Status LED
led = None
wlan = None
mqtt_client = None

# Data collection
current_pattern = 0
cosmic_count = 0
spi = None
i2c = None
uart = None
bmp280 = None
gps = None

# Main data structure
data = {
    "timestamp": 0,
    "device_id": DEVICE_ID,
    "uptime_seconds": 0,
    "chip_count": CONFIG["sram_chips"],
    "pattern": CONFIG["test_pattern"][0],
    "seu_count": 0,
    "seu_by_chip": [0] * CONFIG["sram_chips"],
    "seu_by_pattern": [0] * len(CONFIG["test_pattern"]),
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
    "battery_voltage": 0,
    "version": VERSION,
}

# Time tracking
start_time = 0
last_log_time = 0
last_sensor_read = 0
last_sram_check = 0
last_hb_time = 0

def blink_led(times=1, delay=0.1):
    """Blink the LED to indicate status"""
    if led is None:
        return
    for _ in range(times):
        led.value(1)
        time.sleep(delay)
        led.value(0)
        time.sleep(delay)

def setup_wifi():
    """Set up WiFi connection"""
    global wlan
    print("Setting up WiFi...")
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not CONFIG["wifi_ssid"]:
        print("WiFi credentials not configured!")
        return False
    
    if not wlan.isconnected():
        print(f"Connecting to {CONFIG['wifi_ssid']}...")
        wlan.connect(CONFIG["wifi_ssid"], CONFIG["wifi_password"])
        
        # Wait for connection with timeout
        max_wait = 10
        while max_wait > 0:
            if wlan.status() < 0 or wlan.status() >= 3:
                break
            max_wait -= 1
            print("Waiting for connection...")
            time.sleep(1)
            
    if wlan.isconnected():
        print(f"Connected to WiFi: {CONFIG['wifi_ssid']}")
        print(f"IP address: {wlan.ifconfig()[0]}")
        blink_led(3, 0.1)  # 3 quick blinks for success
        return True
    else:
        print("WiFi connection failed!")
        print(f"Status: {wlan.status()}")
        blink_led(5, 0.5)  # 5 slow blinks for failure
        return False

def setup_mqtt():
    """Set up MQTT client"""
    global mqtt_client
    if not HAVE_MQTT or not CONFIG["use_mqtt"]:
        return False

    try:
        client_id = f"pico-seu-{binascii.hexlify(machine.unique_id()).decode()}"
        mqtt_client = MQTTClient(
            client_id, 
            CONFIG["mqtt_broker"],
            port=CONFIG["mqtt_port"]
        )
        mqtt_client.connect()
        print(f"Connected to MQTT broker: {CONFIG['mqtt_broker']}")
        return True
    except Exception as e:
        print(f"MQTT setup failed: {e}")
        return False

def setup_hardware():
    """Set up the hardware interfaces"""
    global spi, i2c, uart, bmp280, gps, led
    
    if CONFIG["simulation_mode"]:
        print("Running in simulation mode - not initializing hardware")
        return True
    
    try:
        # Set up LED
        led = Pin(CONFIG["led_pin"], Pin.OUT)
        blink_led(2)  # 2 blinks at startup
        
        # Set up SPI for SRAM
        # Pico defaults: sck=6, mosi=7, miso=4
        spi = SPI(0, baudrate=4000000, polarity=0, phase=0)
        
        # Set up chip select pins
        for pin in CONFIG["cs_pins"][:CONFIG["sram_chips"]]:
            cs = Pin(pin, Pin.OUT)
            cs.value(1)  # CS inactive HIGH
        
        # Set up I2C for BMP280
        # Pico defaults: scl=1, sda=0
        i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
        
        # Scan for I2C devices
        devices = i2c.scan()
        print("I2C devices found:", [hex(d) for d in devices])
        
        # Try to detect BMP280 (default address 0x76 or 0x77)
        if 0x76 in devices or 0x77 in devices:
            addr = 0x76 if 0x76 in devices else 0x77
            try:
                import bmp280_lib
                bmp280 = bmp280_lib.BMP280(i2c, addr)
                print("BMP280 detected!")
            except ImportError:
                print("BMP280 library not found")
                
                # Try basic I2C communication
                # BMP280 register 0xD0 is the chip ID register
                try:
                    chip_id = i2c.readfrom_mem(addr, 0xD0, 1)[0]
                    print(f"BMP280 chip ID: {chip_id}")
                    # If chip_id is 0x58, it's a BMP280
                    if chip_id == 0x58:
                        print("BMP280 confirmed!")
                except Exception as e:
                    print(f"BMP280 test failed: {e}")
        else:
            print("BMP280 not detected on I2C bus")
        
        # Set up UART for GPS
        # Pico defaults: tx=4, rx=5
        uart = UART(1, baudrate=9600, tx=Pin(4), rx=Pin(5))
        
        # Set up cosmic ray counter pin with interrupt
        cosmic_pin = Pin(CONFIG["cosmic_counter_pin"], Pin.IN, Pin.PULL_DOWN)
        cosmic_pin.irq(trigger=Pin.IRQ_RISING, handler=cosmic_ray_callback)
        
        # Set up ADC for temperature and battery voltage
        temp_sensor = ADC(4)  # Internal temperature sensor
        battery_adc = ADC(3)  # Use ADC pin for battery monitoring
        
        print("Hardware initialization complete!")
        return True
        
    except Exception as e:
        print(f"Hardware setup failed: {e}")
        return False

def cosmic_ray_callback(pin):
    """Callback for cosmic ray detector pin"""
    global cosmic_count
    cosmic_count += 1

def read_sram(chip, address, nbytes=1):
    """Read data from SRAM chip"""
    if CONFIG["simulation_mode"] or spi is None:
        return [0] * nbytes
    
    try:
        # Select the chip
        cs = Pin(CONFIG["cs_pins"][chip], Pin.OUT)
        cs.value(0)  # Active LOW
        
        # Format SRAM read command
        cmd = bytearray([0x03])  # READ command
        cmd.extend([(address >> 16) & 0xFF, (address >> 8) & 0xFF, address & 0xFF])
        
        # Send command
        spi.write(cmd)
        
        # Read data
        result = bytearray(nbytes)
        spi.readinto(result)
        
        # Deselect chip
        cs.value(1)
        
        return result
    except Exception as e:
        print(f"SRAM read error: {e}")
        return bytearray([0] * nbytes)

def write_sram(chip, address, data):
    """Write data to SRAM chip"""
    if CONFIG["simulation_mode"] or spi is None:
        return
    
    try:
        # Select the chip
        cs = Pin(CONFIG["cs_pins"][chip], Pin.OUT)
        cs.value(0)  # Active LOW
        
        # Format SRAM write command
        cmd = bytearray([0x02])  # WRITE command
        cmd.extend([(address >> 16) & 0xFF, (address >> 8) & 0xFF, address & 0xFF])
        
        # Send command
        spi.write(cmd)
        
        # Write data
        spi.write(bytearray(data))
        
        # Deselect chip
        cs.value(1)
    except Exception as e:
        print(f"SRAM write error: {e}")

def check_sram_errors():
    """Check SRAM for errors against expected pattern"""
    global current_pattern
    
    if CONFIG["simulation_mode"]:
        # Simulate SEU events based on altitude
        altitude = data["altitude_m"]
        import random
        
        # Simulate based on altitude - higher altitude = more SEUs
        if altitude < 100:  # Ground level
            data["seu_count"] = 0
            data["max_run_length"] = 0
            data["seu_by_chip"] = [0] * CONFIG["sram_chips"]
        else:
            # Higher altitude = more SEUs
            base_probability = min(1.0, altitude / 50000)
            seu_count = int(random.random() * base_probability * 10)
            data["seu_count"] = seu_count
            
            if seu_count > 0:
                data["max_run_length"] = min(seu_count, int(random.random() * 2) + 1)
                # Distribute SEUs across chips
                data["seu_by_chip"] = [
                    int(random.random() * seu_count * 0.5) for _ in range(CONFIG["sram_chips"])
                ]
            else:
                data["max_run_length"] = 0
                data["seu_by_chip"] = [0] * CONFIG["sram_chips"]
        return
    
    pattern = CONFIG["test_pattern"][current_pattern]
    
    # Change to next pattern for next check
    current_pattern = (current_pattern + 1) % len(CONFIG["test_pattern"])
    
    if spi is None:
        return
    
    try:
        total_bit_flips = 0
        max_consecutive = 0
        current_consecutive = 0
        
        # Check each chip
        for chip in range(CONFIG["sram_chips"]):
            chip_flips = 0
            
            # Check a sample of the SRAM
            sram_size = CONFIG["sram_size_bytes"]
            sample_size = min(1000, sram_size // 10)  # Check ~10% of memory
            
            # Generate random sample addresses
            sample_addrs = []
            for _ in range(sample_size):
                sample_addrs.append(int(random.random() * sram_size))
            
            for addr in sample_addrs:
                # Read byte
                data_byte = read_sram(chip, addr, 1)[0]
                
                # Check for bit flips
                bit_flips = bin(data_byte ^ pattern).count('1')
                total_bit_flips += bit_flips
                chip_flips += bit_flips
                
                # Track consecutive errors
                if bit_flips > 0:
                    current_consecutive += 1
                else:
                    current_consecutive = 0
                
                max_consecutive = max(max_consecutive, current_consecutive)
                
                # Reset pattern for next check
                if bit_flips > 0:
                    write_sram(chip, addr, [pattern])
            
            # Update chip specific data
            data["seu_by_chip"][chip] = chip_flips
        
        # Update overall data
        data["seu_count"] = total_bit_flips
        data["max_run_length"] = max_consecutive
        data["pattern"] = pattern
        
        # Update pattern-specific data
        data["seu_by_pattern"][current_pattern] = total_bit_flips
    
    except Exception as e:
        print(f"SRAM error check failed: {e}")

def read_gps():
    """Read GPS data"""
    if CONFIG["simulation_mode"] or uart is None:
        # Simulated position - generates values that change slightly over time
        import random
        import math
        
        t = time.time() % 3600  # cycle every hour
        
        # Generate simulated GPS data - circular pattern
        radius = 0.01  # approximately 1 km
        center_lat = 37.7749  # San Francisco
        center_lon = -122.4194
        
        angle = (t / 3600) * 2 * math.pi
        lat = center_lat + radius * math.sin(angle)
        lon = center_lon + radius * math.cos(angle)
        
        # Simulated altitude changing over time (sine wave)
        altitude = 100 + 50 * math.sin(t / 600 * math.pi)  # 10-minute cycle
        
        # Add small random variations
        lat += (random.random() - 0.5) * 0.0001
        lon += (random.random() - 0.5) * 0.0001
        altitude += (random.random() - 0.5) * 5
        
        # Speed also varies
        speed = 20 + 10 * math.sin(t / 600 * math.pi)  # km/h
        
        data["latitude"] = lat
        data["longitude"] = lon
        data["altitude_m"] = altitude
        data["speed_kmh"] = speed
        data["satellites"] = 8 + int(random.random() * 6)  # 8-13 satellites
        
        return
    
    # Real GPS reading from UART
    if uart.any():
        line = uart.readline()
        if line:
            try:
                line = line.decode('utf-8').strip()
                if line.startswith('$GPGGA'):
                    # Global Positioning System Fix Data
                    parts = line.split(',')
                    if len(parts) >= 10 and parts[2] and parts[4]:
                        # Extract latitude
                        lat_deg = float(parts[2][:2])
                        lat_min = float(parts[2][2:])
                        lat = lat_deg + (lat_min / 60.0)
                        if parts[3] == 'S':
                            lat = -lat
                            
                        # Extract longitude
                        lon_deg = float(parts[4][:3])
                        lon_min = float(parts[4][3:])
                        lon = lon_deg + (lon_min / 60.0)
                        if parts[5] == 'W':
                            lon = -lon
                            
                        # Update GPS data
                        data["latitude"] = lat
                        data["longitude"] = lon
                        data["satellites"] = int(parts[7]) if parts[7] else 0
                        
                        # Extract altitude if available
                        if len(parts) >= 10 and parts[9]:
                            data["altitude_m"] = float(parts[9])
                            
                elif line.startswith('$GPRMC'):
                    # Recommended Minimum Specific GPS/Transit Data
                    parts = line.split(',')
                    if len(parts) >= 8 and parts[7]:
                        # Extract speed in knots, convert to km/h
                        if parts[7]:
                            speed_knots = float(parts[7])
                            data["speed_kmh"] = speed_knots * 1.852
            except Exception as e:
                print(f"GPS parse error: {e}")

def read_bmp280():
    """Read BMP280 sensor data"""
    if CONFIG["simulation_mode"]:
        # Simulated temperature and pressure
        import random
        import math
        
        t = time.time() % 86400  # cycle every day
        hour = (t / 3600) % 24
        
        # Simulate temperature with a daily cycle + random variation
        # Coolest at 5am, warmest at 3pm
        hour_temp_offset = math.sin((hour - 5) * math.pi / 10) * 10  # -10 to +10
        base_temp = 20 + hour_temp_offset  # Base around 20°C
        data["temperature_c"] = base_temp + (random.random() - 0.5) * 2  # ±1°C random variation
        
        # Simulate pressure with some correlation to altitude
        # Higher altitude = lower pressure
        alt_pressure_factor = data["altitude_m"] / 8.3  # Rough approximation
        data["pressure_hpa"] = 1013.25 - alt_pressure_factor + (random.random() - 0.5) * 2
        
        return
    
    if bmp280 is None:
        return
    
    try:
        # Try to read from bmp280 library
        if hasattr(bmp280, "read_temperature"):
            data["temperature_c"] = bmp280.read_temperature()
            data["pressure_hpa"] = bmp280.read_pressure() / 100  # Pa to hPa
            
            # Calculate altitude using standard formula
            data["altitude_m"] = 44330 * (1 - (data["pressure_hpa"] / 1013.25) ** 0.1903)
        else:
            # If we have direct I2C access but no full library
            # Read raw temperature and pressure from registers
            # This is a simplified version, not as accurate
            try:
                # Get BMP280 address from scan
                devices = i2c.scan()
                bmp_addr = 0x76 if 0x76 in devices else 0x77
                
                # Wake up the sensor (if needed)
                i2c.writeto_mem(bmp_addr, 0xF4, bytes([0x57]))  # Config register
                time.sleep_ms(10)  # Wait for measurement
                
                # Read raw temperature (registers 0xFA-0xFC)
                temp_raw = i2c.readfrom_mem(bmp_addr, 0xFA, 3)
                temp_raw = (temp_raw[0] << 16) | (temp_raw[1] << 8) | temp_raw[2]
                temp_raw = temp_raw >> 4
                
                # Convert to temperature (very simplified)
                # This is just an approximation without the calibration data
                data["temperature_c"] = (temp_raw / 100) - 5  # Rough approximation
                
                # Read raw pressure (registers 0xF7-0xF9)
                pres_raw = i2c.readfrom_mem(bmp_addr, 0xF7, 3)
                pres_raw = (pres_raw[0] << 16) | (pres_raw[1] << 8) | pres_raw[2]
                pres_raw = pres_raw >> 4
                
                # Convert to pressure (very simplified)
                data["pressure_hpa"] = pres_raw / 100  # Rough approximation
            except Exception as e:
                print(f"BMP280 I2C read error: {e}")
    except Exception as e:
        print(f"BMP280 read error: {e}")

def read_system_data():
    """Read system data like temperature and battery voltage"""
    # Internal temperature sensor on Pico
    try:
        temp_sensor = ADC(4)  # Internal temperature sensor
        reading = temp_sensor.read_u16()
        voltage = reading * 3.3 / 65535
        # Temperature calculation from Pico datasheet
        data["cpu_temp_c"] = 27 - (voltage - 0.706) / 0.001721
    except Exception:
        pass
    
    # Battery voltage
    try:
        # If you've connected a battery to an ADC pin, read it
        battery_adc = ADC(3)  # Use appropriate ADC pin
        reading = battery_adc.read_u16()
        # Adjust the scaling based on your voltage divider
        data["battery_voltage"] = reading * 3.3 / 65535 * 2  # Assuming 1:1 voltage divider
    except Exception:
        # Use a default/dummy value
        data["battery_voltage"] = 3.3

def heartbeat():
    """Send a heartbeat signal - blink LED"""
    global last_hb_time
    
    # Heartbeat every 5 seconds
    if time.time() - last_hb_time >= 5:
        blink_led(1, 0.05)  # Quick blink
        last_hb_time = time.time()

def read_sensors():
    """Read all sensor data"""
    global last_sensor_read
    
    # Update timestamp and uptime
    data["timestamp"] = time.time()
    data["uptime_seconds"] = time.time() - start_time
    data["cosmic_counts"] = cosmic_count
    
    # Read all sensors
    read_bmp280()
    read_gps()
    read_system_data()
    
    # Mark last read time
    last_sensor_read = time.time()

def send_data():
    """Send data to the Raspberry Pi Zero 2 W"""
    global last_log_time
    
    # Only send at the specified interval
    if time.time() - last_log_time < CONFIG["log_interval"]:
        return
    
    if not wlan or not wlan.isconnected():
        print("WiFi not connected - cannot send data")
        blink_led(2, 0.5)  # 2 slow blinks for error
        return
    
    data_json = json.dumps(data)
    sent = False
    
    # Try MQTT first if enabled
    if HAVE_MQTT and CONFIG["use_mqtt"] and mqtt_client:
        try:
            mqtt_client.publish(CONFIG["mqtt_topic"], data_json)
            print(f"MQTT data sent: {data_json[:60]}...")
            sent = True
        except Exception as e:
            print(f"MQTT send error: {e}")
            # Try to reconnect
            try:
                mqtt_client.connect()
                mqtt_client.publish(CONFIG["mqtt_topic"], data_json)
                print("MQTT reconnected and data sent")
                sent = True
            except Exception:
                print("MQTT reconnection failed")
    
    # Fall back to HTTP if MQTT fails or is disabled
    if not sent and HAVE_REQUESTS:
        try:
            response = urequests.post(
                CONFIG["http_server"],
                headers={"Content-Type": "application/json"},
                data=data_json
            )
            print(f"HTTP response: {response.status_code}")
            response.close()
            sent = True
        except Exception as e:
            print(f"HTTP send error: {e}")
    
    # Mark time of last log
    last_log_time = time.time()
    
    # Signal success or failure
    if sent:
        blink_led(1, 0.1)  # Quick blink for success
    else:
        blink_led(3, 0.2)  # 3 blinks for failure

def main():
    """Main program loop"""
    global start_time, last_log_time, last_sensor_read, last_sram_check
    
    print("\nRaspberry Pi Pico W SEU Detector")
    print(f"Version: {VERSION}")
    
    # Initialize hardware
    setup_hardware()
    
    # Initialize SRAM pattern
    if not CONFIG["simulation_mode"]:
        print("Initializing SRAM with test pattern...")
        pattern = CONFIG["test_pattern"][current_pattern]
        for chip in range(CONFIG["sram_chips"]):
            # Only initialize a portion for speed
            for addr in range(0, CONFIG["sram_size_bytes"], 1024):
                write_sram(chip, addr, [pattern])
            print(f"Chip {chip} initialized with pattern {hex(pattern)}")
    
    # Set up networking
    wifi_ok = setup_wifi()
    
    if wifi_ok and CONFIG["use_mqtt"]:
        setup_mqtt()
    
    # Initialize timing
    start_time = time.time()
    last_log_time = time.time()
    last_sensor_read = time.time()
    last_sram_check = time.time()
    last_hb_time = time.time()
    
    # Sample and log intervals
    sample_interval = 1.0 / CONFIG["sample_rate"]
    sram_check_interval = 1.0  # Check SRAM every second
    
    print("Starting main loop...")
    
    try:
        while True:
            current_time = time.time()
            
            # Heartbeat for status indication
            heartbeat()
            
            # Check if it's time to read sensors
            if current_time - last_sensor_read >= sample_interval:
                read_sensors()
            
            # Check if it's time to check SRAM
            if current_time - last_sram_check >= sram_check_interval:
                check_sram_errors()
                last_sram_check = current_time
            
            # Check if it's time to send data
            send_data()
            
            # Collect garbage to prevent memory issues
            if current_time - last_log_time > 30:  # Every 30 seconds
                gc.collect()
            
            # Sleep to save power
            time.sleep_ms(10)
            
    except KeyboardInterrupt:
        print("Program stopped by user")
    finally:
        # Clean up
        if mqtt_client:
            mqtt_client.disconnect()
        print("Program ended")

if __name__ == "__main__":
    main()
