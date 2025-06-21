#!/bin/bash
# Verify SEU Detector Auto-Start Script
# This script verifies that the SEU detector will start automatically on power-up.

# ANSI color codes for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== SEU Detector Auto-Start Verification ===${NC}"
echo "This script verifies that the SEU detector will start automatically when power is applied."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
  echo -e "${RED}Please run as root: sudo ./verify_auto_start.sh${NC}"
  exit 1
fi

echo -e "\n${YELLOW}1. Checking if SEU detector service is installed...${NC}"
if [ -f "/etc/systemd/system/seu-detector.service" ]; then
  echo -e "${GREEN}✓ SEU detector service is installed.${NC}"
else
  echo -e "${RED}✗ SEU detector service is NOT installed.${NC}"
  echo "Run the setup script first: sudo bash setup_pi_zero2w_single.sh"
  exit 1
fi

echo -e "\n${YELLOW}2. Checking if service is enabled to start on boot...${NC}"
ENABLED=$(systemctl is-enabled seu-detector.service 2>/dev/null)
if [ "$ENABLED" = "enabled" ]; then
  echo -e "${GREEN}✓ SEU detector service is enabled for auto-start.${NC}"
else
  echo -e "${RED}✗ SEU detector service is NOT enabled for auto-start.${NC}"
  echo "Enabling service..."
  systemctl enable seu-detector.service
  
  # Check again
  ENABLED=$(systemctl is-enabled seu-detector.service 2>/dev/null)
  if [ "$ENABLED" = "enabled" ]; then
    echo -e "${GREEN}✓ Service has been enabled.${NC}"
  else
    echo -e "${RED}✗ Failed to enable service. Manual configuration required.${NC}"
    exit 1
  fi
fi

echo -e "\n${YELLOW}3. Checking current service status...${NC}"
STATUS=$(systemctl is-active seu-detector.service)
if [ "$STATUS" = "active" ]; then
  echo -e "${GREEN}✓ SEU detector service is currently running.${NC}"
else
  echo -e "${YELLOW}! SEU detector service is not currently running.${NC}"
  echo "Starting service..."
  systemctl start seu-detector.service
  sleep 3
  
  # Check again
  STATUS=$(systemctl is-active seu-detector.service)
  if [ "$STATUS" = "active" ]; then
    echo -e "${GREEN}✓ Service has been started.${NC}"
  else
    echo -e "${RED}✗ Failed to start service. Check logs with: journalctl -u seu-detector.service${NC}"
  fi
fi

echo -e "\n${YELLOW}4. Checking if data collection is working...${NC}"
sleep 2
# Get the last 5 log entries and check for any errors
LOG_ERRORS=$(journalctl -u seu-detector.service -n 5 | grep -i "error\|failure\|failed" | wc -l)
if [ "$LOG_ERRORS" -gt 0 ]; then
  echo -e "${RED}✗ Found potential errors in the logs:${NC}"
  journalctl -u seu-detector.service -n 5 | grep -i "error\|failure\|failed"
else
  echo -e "${GREEN}✓ No immediate errors detected in logs.${NC}"
fi

echo -e "\n${YELLOW}=== Auto-Start Test Results ===${NC}"
if [ "$ENABLED" = "enabled" ] && [ "$STATUS" = "active" ] && [ "$LOG_ERRORS" -eq 0 ]; then
  echo -e "${GREEN}✓ SEU detector is properly configured to start automatically on power-up.${NC}"
  echo -e "${GREEN}✓ Service is currently active and running.${NC}"
  echo ""
  echo -e "${YELLOW}For a complete verification, follow these steps:${NC}"
  echo "1. Shut down: sudo shutdown -h now"
  echo "2. Disconnect power completely"
  echo "3. Reconnect power and wait 2-3 minutes"
  echo "4. Check that service started: systemctl status seu-detector.service"
else
  echo -e "${RED}✗ There are issues with the SEU detector auto-start configuration.${NC}"
  echo "Please review the errors above and fix them before flight."
fi

echo ""
echo "Log details (last 10 entries):"
echo "-----------------------------"
journalctl -u seu-detector.service -n 10
echo "-----------------------------"

echo -e "\n${YELLOW}Memory usage:${NC}"
free -h

echo -e "\n${YELLOW}Disk space:${NC}"
df -h | grep -E "Filesystem|/$"
