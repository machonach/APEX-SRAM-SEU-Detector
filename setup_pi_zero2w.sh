#!/bin/bash
# Script to set up the Raspberry Pi Zero 2 W for the SEU Detector project
# Run with sudo: sudo bash setup_pi_zero2w.sh

# Exit on error
set -e

echo "==== Setting up SEU Detector on Raspberry Pi Zero 2 W ===="
echo "This script will install all necessary components for the SEU Detector"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root: sudo bash $0"
  exit 1
fi

# Update the system
echo "Updating system packages..."
apt update && apt upgrade -y

# Install required packages
echo "Installing required packages..."
apt install -y python3-pip git python3-venv i2c-tools libatlas-base-dev

# Enable interfaces
echo "Enabling required interfaces..."
echo "You may need to manually enable SPI, I2C, and Serial using raspi-config"
echo "Run: sudo raspi-config"
echo "Navigate to: Interfacing Options > Enable SPI, I2C, Serial"

# Create Python virtual environment and install dependencies
echo "Setting up Python environment..."
cd /home/pi

# Check if project directory exists
if [ ! -d "APEX-SRAM-SEU-Detector" ]; then
    echo "Cloning project repository..."
    sudo -u pi git clone https://github.com/machonach/APEX-SRAM-SEU-Detector.git
else
    echo "Project directory exists. Updating repository..."
    cd APEX-SRAM-SEU-Detector
    sudo -u pi git pull
    cd ..
fi

cd APEX-SRAM-SEU-Detector

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    sudo -u pi python3 -m venv venv
fi

# Install requirements
echo "Installing Python dependencies..."
sudo -u pi venv/bin/pip install -r requirements.txt

# Create systemd service for the SEU detector
echo "Creating systemd service for the SEU detector..."
cat > /etc/systemd/system/seu-detector.service << EOF
[Unit]
Description=SEU Detector Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/APEX-SRAM-SEU-Detector
ExecStart=/home/pi/APEX-SRAM-SEU-Detector/venv/bin/python3 raspberry_pi_seu_detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
echo "Enabling service..."
systemctl daemon-reload
systemctl enable seu-detector.service

echo ""
echo "Setup complete! The SEU detector service has been installed."
echo ""
echo "To control the service:"
echo "  Start:   sudo systemctl start seu-detector"
echo "  Stop:    sudo systemctl stop seu-detector"
echo "  Status:  systemctl status seu-detector"
echo "  Logs:    journalctl -u seu-detector -f"
echo ""
echo "For high-altitude launch mode, run:"
echo "  python3 high_altitude_mode.py --enable"
echo ""
echo "Happy data collection!"
