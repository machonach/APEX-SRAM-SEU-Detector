# High-Altitude Balloon Launch Guide

This guide provides detailed instructions for preparing and launching the SEU detector system on a high-altitude balloon, targeting altitudes up to 103,000 feet (31.4 km).

## Pre-Flight Checklist

### Components
- [ ] Raspberry Pi Zero 2 W with SD card
- [ ] SRAM chips (23LC1024 or similar)
- [ ] BMP280 temperature/pressure sensor
- [ ] GPS module (high-altitude capable)
- [ ] Power supply (LiPo batteries with voltage regulators)
- [ ] Insulated enclosure
- [ ] High-altitude balloon
- [ ] Parachute
- [ ] Cord/line
- [ ] Hand warmers (for extreme cold)
- [ ] Tracking devices

### Physical Setup
- [ ] All components securely mounted
- [ ] Wiring properly connected and secured
- [ ] Batteries fully charged
- [ ] Insulation properly applied
- [ ] Heating solution in place
- [ ] Recovery information (phone number, reward offer) on external surface

### Software Setup
- [ ] Latest software installed on Pi Zero 2 W
- [ ] High-altitude mode configured
- [ ] Offline data storage tested
- [ ] System boots properly when power is applied
- [ ] Auto-start on boot confirmed

## Environmental Considerations

### Temperature
At 103,000 feet (~31 km), temperatures can reach -60°C to -70°C (-76°F to -94°F). Your system needs to:
- Be well insulated (styrofoam or similar material)
- Have battery heating solutions
- Use components rated for low temperatures

### Air Pressure
At maximum altitude, pressure is approximately 1% of sea level. This means:
- Sealed containers may expand and rupture
- Some electronics may behave differently
- Heat dissipation is reduced (convection cooling less effective)

### Radiation
Higher radiation levels will be perfect for SEU detection, but be aware:
- Data storage media should be radiation-resistant if possible
- Include multiple redundant storage methods

## Launch Preparation

### 1. Final System Testing (24-48 hours before launch)
```bash
# On the Pi Zero 2 W
cd APEX-SRAM-SEU-Detector
chmod +x high_altitude_mode.py
python3 high_altitude_mode.py --enable --auto-start --config custom_launch_config.json

# CRITICAL: Verify auto-start is properly enabled
systemctl is-enabled seu-detector  # Should output "enabled"

# Test a reboot to ensure it starts automatically
sudo reboot
# (wait for Pi to reboot, then reconnect)
systemctl status seu-detector  # Should show "active (running)"
```

### 2. Power Management Setup
- Calculate expected power consumption and flight duration
- Set up proper voltage regulation
- Ensure stable power to all components

### 3. Data Storage Preparation
```bash
# On Pi Zero 2 W
# Format SD card and ensure plenty of free space
df -h
sudo mkdir -p /data/seu_flight
sudo chmod 777 /data/seu_flight
```

### 4. Final Software Preparation
- Update the configuration file for flight mode
- Enable all sensors
- Set proper sampling rates
- Enable offline storage

### 5. Power-Up Auto-Start
- Confirm the systemd service (seu-detector.service) is enabled
- The `--auto-start` flag ensures the detector starts immediately when power is applied
- **Test auto-start by disconnecting and reconnecting power** to simulate launch conditions
- Watch the logs after power-up to verify immediate startup: `journalctl -u seu-detector -f`

## Power-On Test Procedure

To ensure the system will start automatically during the balloon launch when power is applied, follow this test procedure:

1. Configure high-altitude mode with auto-start:
```bash
cd APEX-SRAM-SEU-Detector
python3 high_altitude_mode.py --enable --auto-start
```

2. Verify the service is enabled:
```bash
systemctl is-enabled seu-detector  # Should output "enabled"
```

3. **Simulate a power cycle:**
   - Shut down the Pi: `sudo shutdown -h now`
   - After shutdown is complete, disconnect power
   - Wait 30 seconds
   - Reconnect power

4. Wait 2-3 minutes for the Pi to boot completely

5. Connect to the Pi and verify the service started automatically:
```bash
systemctl status seu-detector
```

6. Confirm data is being collected:
```bash
journalctl -u seu-detector -n 20
```

7. Check data files are being saved:
```bash
ls -la ~/APEX-SRAM-SEU-Detector/data/
```

If all steps pass, your system is configured correctly for autonomous operation during the balloon flight.

> **CRITICAL FLIGHT CHECK:** For extra certainty, always perform one final test with the *actual* flight battery that will be used during launch.

## Launch Day Procedures

### 1. Pre-Launch (2-3 hours before)
- Power up the system
- Verify all sensors are working
- Check GPS lock
- Verify data is being recorded
- Check battery levels

### 2. Final Checks (30 minutes before)
- Secure all components in the payload container
- Verify system is in high-altitude mode
- Ensure SD card has sufficient space
- Verify startup services are enabled
- Perform final hardware check

### 3. Launch Procedures
- Carefully attach payload to balloon/parachute assembly
- Ensure no wires or components can snag
- Power on the system one final time
- Verify status lights indicate proper operation
- Launch according to balloon flight protocols

## Recovery Procedures

### 1. Tracking
- Use GPS tracking if available
- Consider adding a radio beacon
- Add physical contact information to exterior

### 2. Post-Recovery
- Power down the system properly
- Copy all data before handling
- Check for physical damage
- Backup SD card immediately

### 3. Data Analysis
- Extract data files from SD card
- Follow standard analysis procedures
- Compare with control data if available

## Troubleshooting

### Common Issues
- **Power failure**: Check battery connections and voltage regulators
- **GPS loss**: Ensure antenna has clear view of sky
- **Data recording issues**: Check SD card integrity and free space
- **Cold-related failures**: Add additional insulation and heat packs

## Safety Considerations

- Obtain necessary permits for balloon launches
- Check weather conditions before launch
- Ensure payload is under weight limits
- Follow all local regulations for high-altitude balloons
- Notify relevant aviation authorities if required

## Appendix: Configuration Options

Key configuration values for high-altitude mode:

```json
{
  "sample_rate": 1,              // Readings per second
  "log_interval": 60,            // Log every minute
  "power_saving": true,          // Enable power saving features
  "high_altitude_mode": true,    // Enable high-altitude optimizations
  "offline_mode": true           // Ensure offline data storage
}
```

Good luck with your high-altitude SEU detection mission!
