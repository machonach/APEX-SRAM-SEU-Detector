# Raspberry Pi Pico W + Raspberry Pi Zero 2 W SEU Detector

This is a distributed SEU (Single Event Upset) detection system using a Raspberry Pi Pico W as a data collection node and a Raspberry Pi Zero 2 W as the central processing unit.

## Overview

This setup allows you to:
1. Collect SEU data using the Raspberry Pi Pico W with SRAM chips
2. Monitor environmental conditions with BMP280 sensor
3. Track location with GPS
4. Count cosmic ray events
5. Transmit all data to the Raspberry Pi Zero 2 W for processing and storage

## Hardware Requirements

### Raspberry Pi Pico W
- Raspberry Pi Pico W microcontroller
- SRAM chips (23LC1024 or similar)
- BMP280 temperature/pressure sensor
- GPS module
- Cosmic ray detector (optional)

### Raspberry Pi Zero 2 W
- Raspberry Pi Zero 2 W
- SD card (16GB+ recommended)
- WiFi connection


### Raspberry Pi Pico W Connections

```
- SPI: SRAM Chips
  - GP7 (SPI0 TX/MOSI) -> SRAM SI
  - GP4 (SPI0 RX/MISO) -> SRAM SO
  - GP6 (SPI0 SCK) -> SRAM SCLK
  - GP5 (SPI0 CSn) -> SRAM CS
  - 3.3V -> SRAM VCC
  - GND -> SRAM GND

- I2C: BMP280 Sensor
  - GP0 (I2C0 SDA) -> BMP280 SDA
  - GP1 (I2C0 SCL) -> BMP280 SCL
  - 3.3V -> BMP280 VCC
  - GND -> BMP280 GND

- UART: GPS Module
  - GP4 (UART1 TX) -> GPS RX
  - GP5 (UART1 RX) -> GPS TX
  - 3.3V -> GPS VCC
  - GND -> GPS GND

- GPIO: Cosmic Ray Detector
  - GP22 -> Detector Pulse Output
  - 3.3V -> Detector VCC
  - GND -> Detector GND
```

## Software Setup

### Raspberry Pi Zero 2 W Setup

1. Flash Raspberry Pi OS (Lite is recommended) to the SD card
2. Enable SSH and configure WiFi
3. Log in to the Pi and run the setup script:

```bash
git clone https://github.com/machonach/APEX-SRAM-SEU-Detector.git
cd APEX-SRAM-SEU-Detector
sudo bash setup_pi_zero2w.sh
```

4. The script will install all required dependencies and create systemd services

### Raspberry Pi Pico W Setup

1. Download [MicroPython firmware for Raspberry Pi Pico W](https://micropython.org/download/rp2-pico-w/)
2. Flash MicroPython to the Pico W:
   - Hold the BOOTSEL button while connecting to USB
   - Drag and drop the MicroPython .uf2 file to the RPI-RP2 drive
   
3. Connect to the Pico W using Thonny IDE or rshell
4. Install the required files:
   - Copy `pico_boot.py` to the Pico W as `boot.py`
   - Copy `pico_seu_collector.py` to the Pico W as `main.py`
   - Copy `bmp280_lib.py` to the Pico W

5. Edit the WiFi settings in `main.py`:
   ```python
   CONFIG = {
       # ... other settings ...
       "wifi_ssid": "YOUR_WIFI_SSID",       # Set your WiFi name
       "wifi_password": "YOUR_WIFI_PASSWORD", # Set your WiFi password
       "mqtt_broker": "192.168.1.XXX",      # Set to your Pi Zero 2 W IP
       # ... other settings ...
   }
   ```

## Running the System

### Raspberry Pi Zero 2 W Services

The following services are automatically installed and can be controlled:

```bash
# Start the services
sudo systemctl start mosquitto
sudo systemctl start seu-detector
sudo systemctl start pico-receiver

# Check service status
systemctl status seu-detector
systemctl status pico-receiver

# View logs
journalctl -u seu-detector -f
journalctl -u pico-receiver -f
```

### Data Flow

1. The Pico W collects data from:
   - SRAM chips (checking for bit flips)
   - BMP280 (temperature, pressure, altitude)
   - GPS (location, speed, altitude)
   - Cosmic ray detector

2. Data is transmitted via WiFi using MQTT to the Pi Zero 2 W

3. The Pi Zero 2 W:
   - Receives the data from the Pico W
   - Processes and stores the data
   - Runs the analytics and visualization components
   - Makes data available via API

## Troubleshooting

### Pico W Issues

- **WiFi Connection Failure**: LED will blink 5 slow times if WiFi connection fails
  - Check WiFi credentials in `main.py`
  - Make sure the WiFi network is in range

- **Safe Mode**: If the Pico W isn't responding, connect while holding GP0 to boot into safe mode
  - This prevents `main.py` from running automatically
  - You can connect with REPL and debug

### Raspberry Pi Zero 2 W Issues

- **MQTT Broker not running**: 
  ```bash
  sudo systemctl status mosquitto
  sudo systemctl restart mosquitto
  ```

- **No data being received**:
  - Check that the Pico W is connected to WiFi
  - Verify the correct MQTT broker IP is set in the Pico W code
  - Check the MQTT topic in both the sender and receiver
  ```bash
  # To monitor MQTT messages:
  mosquitto_sub -t "seu/pico/data" -v
  ```

## Extending the System

You can extend this system by:
1. Adding multiple Pico W nodes for distributed data collection
2. Implementing different sensor types
3. Creating a real-time dashboard to monitor all nodes
4. Analyzing data for patterns and correlations between SEUs and environmental factors
