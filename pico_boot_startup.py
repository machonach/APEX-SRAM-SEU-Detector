"""
Boot file for the Raspberry Pi Pico W SEU Detector.
This file runs automatically when the Pico W starts up.
It handles basic setup before running the main program.
"""

import machine
import time
import gc
import os

# Function to handle safe boot (holding GP0 during boot)
def check_safe_boot():
    """Check if the BOOTSEL button is pressed to enter safe mode."""
    button = machine.Pin(15, machine.Pin.IN, machine.Pin.PULL_UP)  # Use a pin connected to BOOTSEL
    if not button.value():
        print("BOOTSEL button pressed - entering safe mode")
        return True
    return False

# Clear the console
print("\n" * 5)
print("="*40)
print("Raspberry Pi Pico W - SEU Detector")
print("="*40)

# Enable garbage collection
gc.enable()
gc.collect()
print(f"Free memory: {gc.mem_free()} bytes")

# Check if we should enter safe mode
safe_mode = check_safe_boot()

if not safe_mode:
    try:
        # Set CPU frequency (higher speed uses more power)
        # Options: 125000000 (125MHz), 100000000 (100MHz), 80000000 (80MHz)
        machine.freq(125_000_000)
        print(f"CPU Frequency set to {machine.freq()/1_000_000}MHz")
        
        # Show system info
        print("\nSystem Information:")
        print(f"MicroPython Version: {os.uname().version}")
        print(f"Machine: {os.uname().machine}")
        print(f"Node Name: {os.uname().nodename}")
        
        # List connected I2C devices (if I2C is available)
        try:
            i2c = machine.I2C(0, sda=machine.Pin(0), scl=machine.Pin(1), freq=400000)
            devices = i2c.scan()
            print("\nI2C Devices found:", [hex(d) for d in devices])
        except Exception as e:
            print(f"I2C scan failed: {e}")
        
        # Blink LED to indicate successful boot
        led = machine.Pin("LED", machine.Pin.OUT)
        for _ in range(3):
            led.value(1)
            time.sleep(0.1)
            led.value(0)
            time.sleep(0.1)
        
        print("\nLoading SEU detector program...")
        print("="*40)
        
        # Import and run the main program
        # The actual import will happen after boot.py finishes
        
    except Exception as e:
        print(f"Boot error: {e}")
        safe_mode = True

if safe_mode:
    print("\nRunning in SAFE MODE")
    print("Only essential services started")
    print("Use REPL to manually fix issues")
    print("="*40)
