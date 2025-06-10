# SEU Detector - Installation and Setup Guide

This guide provides detailed instructions for setting up and running the SRAM SEU (Single Event Upset) Detector system.

## Table of Contents
1. [System Overview](#system-overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Software Requirements](#software-requirements)
4. [Raspberry Pi Setup](#raspberry-pi-setup)
5. [PC/Server Setup](#pcserver-setup)
6. [Running the System](#running-the-system)
7. [Troubleshooting](#troubleshooting)

## System Overview

The SRAM SEU Detector is designed to detect and analyze Single Event Upsets (bit flips) in SRAM memory caused by cosmic radiation. The system consists of:

1. **Data Collection**: Raspberry Pi Zero 2 W with SRAM chips, sensors, and GPS module
2. **Data Analysis**: Machine learning pipeline for analyzing SEU events
3. **Visualization**: Real-time and historical data dashboards
4. **API**: RESTful API for external system integration

## Hardware Requirements

### Raspberry Pi Zero Setup
- Raspberry Pi Zero 2 W
- MicroSD card (16GB+ recommended)
- SRAM chips (23LC1024 or similar)
- BMP280 temperature/pressure sensor
- GPS module (UART interface)
- SiPM or Geiger counter for cosmic ray detection (optional)
- Battery pack (for high-altitude operation)

### Wiring Diagram
```
Raspberry Pi Zero 2 W:
- SPI: SRAM Chips
  - Pin 19 (MOSI) -> SRAM SI
  - Pin 21 (MISO) -> SRAM SO
  - Pin 23 (SCLK) -> SRAM SCLK
  - Pin 24 (CE0) -> SRAM CS
  - 3.3V -> SRAM VCC
  - GND -> SRAM GND

- I2C: BMP280 Sensor
  - Pin 3 (SDA) -> BMP280 SDA
  - Pin 5 (SCL) -> BMP280 SCL
  - 3.3V -> BMP280 VCC
  - GND -> BMP280 GND

- UART: GPS Module
  - Pin 8 (TX) -> GPS RX
  - Pin 10 (RX) -> GPS TX
  - 3.3V -> GPS VCC
  - GND -> GPS GND

- GPIO: Cosmic Ray Detector
  - Pin 18 -> Detector Pulse Output
  - 3.3V -> Detector VCC
  - GND -> Detector GND
```

## Software Requirements

### Raspberry Pi
- Raspberry Pi OS (Lite version recommended for resource efficiency)
- Python 3.9+
- Required Python packages:
  ```
  RPi.GPIO
  spidev
  smbus2
  pyserial
  adafruit-circuitpython-bmp280
  paho-mqtt
  ```

### PC/Server (Data Analysis)
- Python 3.9+
- Required Python packages:
  ```
  numpy
  pandas
  matplotlib
  seaborn
  scikit-learn
  dash
  plotly
  fastapi
  uvicorn
  joblib
  paho-mqtt
  ```

## Raspberry Pi Setup

1. **Install Raspberry Pi OS**:
   - Download Raspberry Pi Imager: https://www.raspberrypi.org/software/
   - Flash Raspberry Pi OS (Lite) to the microSD card
   - Enable SSH and configure WiFi in the imager settings

2. **Connect to the Raspberry Pi**:
   ```bash
   ssh pi@raspberrypi.local
   ```

3. **Update the system**:
   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

4. **Install required packages**:
   ```bash
   sudo apt install -y python3-pip git python3-venv i2c-tools
   sudo pip3 install --upgrade setuptools
   ```

5. **Enable interfaces**:
   ```bash
   sudo raspi-config
   ```
   - Navigate to "Interfacing Options"
   - Enable SPI, I2C, and Serial interfaces

6. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/seu-detector.git
   cd seu-detector
   ```

7. **Set up Python environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements_raspberry_pi.txt
   ```

8. **Test hardware connections**:
   ```bash
   # Test I2C
   sudo i2cdetect -y 1
   
   # Test SPI
   # The command will list SPI devices if properly connected
   ls -l /dev/spidev*
   
   # Test UART/Serial
   # Should show ttyS0 or similar
   ls -l /dev/tty*
   ```

9. **Configure auto-start on boot**:
   ```bash
   sudo nano /etc/systemd/system/seu-detector.service
   ```
   
   Add the following:
   ```ini
   [Unit]
   Description=SEU Detector Service
   After=network.target
   
   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/seu-detector
   ExecStart=/home/pi/seu-detector/venv/bin/python3 raspberry_pi_seu_detector.py
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   Enable the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable seu-detector.service
   sudo systemctl start seu-detector.service
   ```

## PC/Server Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/seu-detector.git
   cd seu-detector
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Configure the system** (optional):
   ```bash
   # Edit configuration files if needed
   nano config.json
   ```

## Running the System

### Data Collection (Raspberry Pi)
Data collection should start automatically on boot if you've set up the systemd service. To manually start:
```bash
python raspberry_pi_seu_detector.py
```

Add `-s` flag to run in simulation mode without hardware:
```bash
python raspberry_pi_seu_detector.py -s
```

### Generate Synthetic Data (PC/Server)
If you don't have real data yet, generate synthetic data:
```bash
cd ml_pipeline
python SEU-Synthetic-Data-Creator.py
```

### Run ML Pipeline
```bash
cd ml_pipeline
python SEU-ML-Pipeline.py
```

### Run Anomaly Detection Training
```bash
cd ml_pipeline
python anomaly_detection.py
```

### Start API Server
```bash
python api_server.py
```
The API will be accessible at http://localhost:8000

### Start Real-time Monitor
```bash
python real_time_monitor.py
```
The dashboard will be accessible at http://localhost:8050

### Start Full Dashboard
```bash
python integrated_dashboard.py
```
The dashboard will be accessible at http://localhost:8050

## Troubleshooting

### Hardware Issues

1. **SRAM not responding**:
   - Check SPI connections
   - Verify that SPI is enabled: `ls -l /dev/spidev*`
   - Test with `spi-test` utility

2. **Temperature sensor not reading**:
   - Verify I2C address: `sudo i2cdetect -y 1`
   - Check power connections
   - Try a different sensor to rule out hardware failure

3. **GPS not providing location**:
   - Check UART connections
   - Give the GPS clear sky view
   - Verify serial port: `ls -l /dev/ttyS0` or `ls -l /dev/ttyAMA0`

4. **Cosmic ray detector not counting**:
   - Check GPIO connections
   - Verify pulse height is sufficient to trigger the GPIO pin
   - Test with a simple GPIO test script

### Software Issues

1. **Raspberry Pi script crashing**:
   - Check logs: `sudo journalctl -u seu-detector.service`
   - Run manually with verbose logging: `python raspberry_pi_seu_detector.py -v`
   - Try simulation mode to verify software functionality: `python raspberry_pi_seu_detector.py -s`

2. **Dashboard not showing data**:
   - Verify that data files exist
   - Check API is running: `curl http://localhost:8000/data/latest`
   - Inspect browser console for JavaScript errors

3. **ML pipeline errors**:
   - Verify input data format
   - Check that all required packages are installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
