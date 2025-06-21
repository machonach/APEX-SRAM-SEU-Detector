# Single-Board SEU Detector Setup Guide

This guide outlines the simplified setup for the Single Event Upset (SEU) Detector system using just a Raspberry Pi Zero 2 W for your high-altitude balloon mission.

## Overview

This project has been streamlined to use only a Raspberry Pi Zero 2 W to collect SRAM, environmental, GPS, and cosmic ray data during a high-altitude balloon flight to 103,000 feet.

## Hardware Requirements

- Raspberry Pi Zero 2 W
- SRAM chips (23LC1024 or similar, quantity: 4)
- BMP280 temperature/pressure sensor
- GPS module (high-altitude capable)
- Cosmic ray detector (optional)
- Power supply (LiPo batteries with voltage regulation)
- Insulated enclosure for high-altitude operation

## Hardware Connection Guide

### SRAM Chips (SPI)
```
- MOSI (GPIO 10) → SRAM SI
- MISO (GPIO 9) → SRAM SO
- SCLK (GPIO 11) → SRAM SCLK
- CE0 (GPIO 8) + Additional CS pins → SRAM CS pins
- 3.3V → SRAM VCC
- GND → SRAM GND
```

### BMP280 (I2C)
```
- SDA (GPIO 2) → BMP280 SDA
- SCL (GPIO 3) → BMP280 SCL
- 3.3V → BMP280 VCC
- GND → BMP280 GND
```

### GPS Module (UART)
```
- TX (GPIO 14) → GPS RX
- RX (GPIO 15) → GPS TX
- 3.3V → GPS VCC
- GND → GPS GND
```

### Cosmic Ray Detector (GPIO)
```
- GPIO 17 → Detector Pulse Output
- 3.3V → Detector VCC
- GND → Detector GND
```

## Software Setup

1. Start with a clean Raspberry Pi OS installation
2. Run the simplified setup script:

```bash
sudo bash setup_pi_zero2w_single.sh
```

This script will:
- Update the system
- Install required dependencies
- Set up the Python environment
- Create a systemd service for the SEU detector

## High-Altitude Configuration

For high-altitude operation, run this **before your flight**:

```bash
python3 high_altitude_mode.py --enable --auto-start
```

This will configure the system for:
- **Guaranteed automatic startup on power-up** (the `--auto-start` flag ensures this)
- Fully offline data storage (no network needed)
- Power saving optimizations
- Optimized sample rates for altitude
- Disabled WiFi to save power and avoid interference

## Automatic Operation

Once configured and powered on, the SEU detector will **automatically start collecting data without any intervention required**. The systemd service ensures the detector starts immediately when power is applied to the Pi. There are two key aspects to this automatic operation:

1. **Immediate startup on power** - The SEU detector will start as soon as the Pi receives power
2. **Fully offline operation** - No network, keyboard, monitor, or user input is needed

### Pre-Flight Check

Before sealing your payload, ALWAYS verify proper operation using the verification script:

```bash
# Make the verification script executable
chmod +x verify_auto_start.sh

# Run the verification script as root
sudo ./verify_auto_start.sh
```

The script will check:
1. If the service is installed
2. If it's enabled for auto-start
3. If it's currently running
4. If there are any errors in the logs

For a complete verification, also perform these additional tests:

```bash
# Simulate a power cycle by shutting down
sudo shutdown -h now
# Physically disconnect power for 30 seconds
# Reconnect power and wait 2-3 minutes
# Log back in and check status
systemctl status seu-detector

# View live data collection to verify sensors are working (CTRL+C to exit)
journalctl -u seu-detector -f
```

> **CRITICAL:** Always test with the actual flight battery before launch to ensure everything works as expected in real conditions.
```

### Manual Control (for testing only)

These commands are only needed during testing, not for the actual flight:

```bash
# Stop the service
sudo systemctl stop seu-detector

# Start the service
sudo systemctl start seu-detector
```

## Launch Preparation

See `HIGH_ALTITUDE_LAUNCH_SINGLE.md` for comprehensive launch preparation guidelines, including:
- Pre-flight checklist
- Environmental considerations
- Launch day procedures
- Recovery instructions

## Why Single-Board?

Using just the Pi Zero 2 W offers several advantages:
- Simplified hardware setup with fewer points of failure
- Reduced power requirements
- Lower weight (important for high-altitude balloons)
- Eliminated communication overhead between devices
- Streamlined data collection pipeline
- Improved reliability in extreme conditions

## Data Analysis

After recovery, data can be found in the configured data directory (default: `/data/seu_flight`). The SEU detector records:
- SEU counts and patterns
- Environmental data (temperature, pressure, altitude)
- GPS location
- Cosmic ray events
- System health metrics

Happy flying and data collecting!
