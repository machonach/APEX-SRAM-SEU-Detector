#!/bin/python3
"""
High-Altitude Launch Mode for SEU Detector
This script configures the SEU Detector system for high-altitude balloon launch
"""

import os
import time
import json
import subprocess
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("launch_mode.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("launch_mode")

# Default configuration
LAUNCH_CONFIG = {
    "enable_wifi": False,          # Disable WiFi to save power during flight
    "enable_led": True,            # Keep LED enabled for status indication  
    "sample_rate": 1,              # 1 Hz sample rate
    "log_interval": 60,            # Log every minute
    "gps_power_save": True,        # Enable GPS power saving
    "max_altitude_feet": 110000,   # Expected maximum altitude
    "expected_flight_hours": 3,    # Expected flight duration
    "auto_shutdown": False,        # Auto shutdown after expected flight time + margin
    "shutdown_delay_hours": 6,     # Hours to wait before auto-shutdown
    "wifi_recovery_mode": True,    # Enable WiFi periodically during descent for recovery
    "wifi_check_interval_min": 15, # Check for ground WiFi every 15 minutes during descent
}

def enable_launch_mode(config=None, enable_autostart=False):
    """Enable launch mode for high-altitude flight"""
    if config is None:
        config = LAUNCH_CONFIG
        
    logger.info("Enabling launch mode for high-altitude flight")
    
    # Disable unnecessary services
    if not config["enable_wifi"]:
        logger.info("Disabling WiFi to save power")
        os.system("rfkill block wifi")
        
    # Disable HDMI if available (for Pi Zero, saves power)
    os.system("/usr/bin/tvservice -o")
    
    # Create a file to indicate we're in launch mode
    with open("/tmp/seu_launch_mode", "w") as f:
        f.write(str(time.time()))
    
    # Update SEU detector configuration
    try:
        seu_config_path = os.path.expanduser("~/APEX-SRAM-SEU-Detector/seu_detector_config.json")
        if os.path.exists(seu_config_path):
            with open(seu_config_path, "r") as f:
                seu_config = json.load(f)
                
            # Update configuration for high altitude
            seu_config["sample_rate"] = config["sample_rate"]
            seu_config["log_interval"] = config["log_interval"]
            seu_config["power_saving"] = True
            seu_config["high_altitude_mode"] = True
            
            with open(seu_config_path, "w") as f:
                json.dump(seu_config, f, indent=2)
                
            logger.info("Updated SEU detector configuration for high altitude")
        else:
            logger.warning("SEU configuration file not found")
    except Exception as e:
        logger.error(f"Failed to update SEU configuration: {e}")
    
    # Set up auto-recovery mode
    if config["wifi_recovery_mode"]:
        # Create a cron job to check for WiFi during descent
        cron_cmd = f"*/15 * * * * /usr/bin/python3 {os.path.abspath(__file__)} --recovery-check"
        with open("/tmp/seu_recovery_cron", "w") as f:
            f.write(cron_cmd)
        os.system("crontab /tmp/seu_recovery_cron")
        logger.info("Set up recovery mode to check for WiFi every 15 minutes")
    
    # Set up auto-shutdown if enabled
    if config["auto_shutdown"]:
        shutdown_seconds = config["shutdown_delay_hours"] * 3600
        logger.info(f"Setting up auto-shutdown after {config['shutdown_delay_hours']} hours")
        os.system(f"shutdown -h +{int(shutdown_seconds/60)}")
    
    # Enable auto-start if requested
    if enable_autostart:
        logger.info("Ensuring SEU detector auto-starts on boot...")
        # Make sure the service is enabled
        autostart_status = os.system("systemctl is-enabled seu-detector.service")
        if autostart_status != 0:
            # Enable the service if it's not already enabled
            os.system("sudo systemctl enable seu-detector.service")
            logger.info("SEU detector service enabled for auto-start on boot")
        else:
            logger.info("SEU detector service is already enabled for auto-start")
        
        # Double-check the service status
        service_status = subprocess.run(["systemctl", "is-enabled", "seu-detector.service"], 
                                        capture_output=True, text=True)
        if "enabled" in service_status.stdout:
            logger.info("Verified: SEU detector will start automatically on power-up")
        else:
            logger.warning("Warning: Auto-start may not be properly configured")
    
    # Restart SEU detector services
    logger.info("Restarting SEU detector services")
    os.system("sudo systemctl restart seu-detector.service")
    
    logger.info("Launch mode enabled. System ready for flight.")
    logger.info(f"Expected maximum altitude: {config['max_altitude_feet']} feet")
    logger.info(f"Expected flight duration: {config['expected_flight_hours']} hours")
    
    return True

def recovery_mode_check():
    """Check if we should enable WiFi for recovery"""
    try:
        # Check if we're in launch mode
        if not os.path.exists("/tmp/seu_launch_mode"):
            return
            
        # Get launch time
        with open("/tmp/seu_launch_mode", "r") as f:
            launch_time = float(f.read().strip())
            
        # Calculate elapsed time
        elapsed_hours = (time.time() - launch_time) / 3600
        
        # If we've passed expected flight duration, start enabling WiFi
        if elapsed_hours >= LAUNCH_CONFIG["expected_flight_hours"]:
            logger.info("Expected flight duration passed, enabling WiFi for recovery")
            os.system("rfkill unblock wifi")
            
            # Try to connect to known WiFi networks
            os.system("wpa_cli reconfigure")
            
            # Keep WiFi on for 5 minutes
            time.sleep(300)
            
            # Turn WiFi off again to save power
            if LAUNCH_CONFIG["enable_wifi"] == False:
                os.system("rfkill block wifi")
    except Exception as e:
        logger.error(f"Recovery mode check failed: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="High-Altitude Launch Mode for SEU Detector")
    parser.add_argument("--enable", action="store_true", help="Enable launch mode")
    parser.add_argument("--disable", action="store_true", help="Disable launch mode")
    parser.add_argument("--recovery-check", action="store_true", help="Check for recovery mode")
    parser.add_argument("--config", help="Path to custom launch configuration file")
    parser.add_argument("--auto-start", action="store_true", help="Ensure system auto-starts on power-up (recommended for flight)")
    args = parser.parse_args()
    
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load custom configuration: {e}")
            config = LAUNCH_CONFIG
    else:
        config = LAUNCH_CONFIG
    
    if args.enable:
        enable_launch_mode(config, enable_autostart=args.auto_start)
    elif args.disable:
        # Disable launch mode
        logger.info("Disabling launch mode")
        os.system("rfkill unblock wifi")
        os.system("/usr/bin/tvservice -p")
        os.system("crontab -r")  # Remove cron job
        if os.path.exists("/tmp/seu_launch_mode"):
            os.remove("/tmp/seu_launch_mode")
        os.system("sudo systemctl restart seu-detector.service")
        logger.info("Launch mode disabled")
    elif args.recovery_check:
        recovery_mode_check()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
