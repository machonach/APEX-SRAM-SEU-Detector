#!/bin/bash
# Script to set up the Raspberry Pi Zero 2 W for the SEU Detector project
# Run with sudo: sudo bash setup_pi_zero2w_single.sh

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

# Install requirements with optimizations
echo "Installing Python dependencies..."
echo "Note: This may take 10-15 minutes on a Pi Zero 2 W. The spinning symbols (/-|\) indicate progress."

# Function to show a spinner animation
show_spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while ps -p $pid > /dev/null; do
        local temp=${spinstr#?}
        printf " [%c] " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Use a timeout and retry mechanism for more reliable installations
for i in {1..3}; do
    echo "==== Installation attempt $i of 3 ===="
    echo "Step 1: Updating pip and setuptools (may take a few minutes)..."
    (sudo -u pi venv/bin/pip install --upgrade pip setuptools wheel) &
    show_spinner $!
    
    echo "Step 2: Installing requirements (this is the long part, please be patient)..."
    # Use a faster PyPI mirror, disable build isolation for speed, and use binary wheels when available
    (sudo -u pi venv/bin/pip install --timeout 180 --prefer-binary --use-feature=fast-deps -r requirements.txt) &
    show_spinner $!
    
    # Check if installation was successful
    if sudo -u pi venv/bin/python -c "import RPi.GPIO; print('Installation successful!')"; then
        echo "✓ Package installation completed successfully!"
        break
    else
        echo "✗ Package installation failed, retrying in 5 seconds..."
        sleep 5
    fi
done

# Create systemd service for the SEU detector
echo "Creating systemd service for the SEU detector..."
cat > /etc/systemd/system/seu-detector.service << EOF
[Unit]
Description=SEU Detector Service
After=multi-user.target
Wants=multi-user.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/APEX-SRAM-SEU-Detector
ExecStart=/home/pi/APEX-SRAM-SEU-Detector/venv/bin/python3 raspberry_pi_seu_detector.py
Restart=always
RestartSec=10
# Make sure the service is restarted even after repeated failures
StartLimitIntervalSec=0

# Optional logging enhancements for easier debugging
StandardOutput=journal
StandardError=journal

[Install]
# Ensure it starts early in the boot process
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
