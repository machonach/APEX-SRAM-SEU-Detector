#!/usr/bin/env python3
"""
SEU Detector - Launcher Script
This script provides a unified way to start different components of the SEU Detector system.
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import time
from pathlib import Path

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

def check_dependencies():
    """Check if required Python packages are installed"""
    try:
        import dash
        import pandas as pd
        import numpy as np
        import plotly
        print("✓ Core dependencies installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        install = input("Would you like to install required dependencies? (y/n): ")
        if install.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                                      os.path.join(PROJECT_ROOT, "requirements.txt")])
                print("✓ Dependencies installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("✗ Failed to install dependencies")
                return False
        return False

def generate_synthetic_data():
    """Generate synthetic data for testing"""
    print("\n== Generating Synthetic Data ==")
    script_path = os.path.join(PROJECT_ROOT, "ml_pipeline", "SEU-Synthetic-Data-Creator.py")
    
    try:
        subprocess.run([sys.executable, script_path], cwd=PROJECT_ROOT, check=True)
        print("✓ Synthetic data generated successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to generate synthetic data")
        return False

def run_ml_pipeline():
    """Run the ML pipeline for data analysis"""
    print("\n== Running ML Pipeline ==")
    script_path = os.path.join(PROJECT_ROOT, "ml_pipeline", "SEU-ML-Pipeline.py")
    
    try:
        subprocess.run([sys.executable, script_path], cwd=PROJECT_ROOT, check=True)
        print("✓ ML pipeline completed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to run ML pipeline")
        return False

def train_anomaly_detection():
    """Train the anomaly detection model"""
    print("\n== Training Anomaly Detection Model ==")
    script_path = os.path.join(PROJECT_ROOT, "ml_pipeline", "anomaly_detection.py")
    
    try:
        subprocess.run([sys.executable, script_path], cwd=PROJECT_ROOT, check=True)
        print("✓ Anomaly detection model trained successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to train anomaly detection model")
        return False

def start_api_server():
    """Start the API server"""
    print("\n== Starting API Server ==")
    script_path = os.path.join(PROJECT_ROOT, "api_server.py")
    
    try:
        # Start process in background
        process = subprocess.Popen([sys.executable, script_path],
                                  cwd=PROJECT_ROOT,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        # Wait a bit for server to start
        time.sleep(2)
        
        # Check if process is still running
        if process.poll() is None:
            print("✓ API server started successfully")
            print("  URL: http://localhost:8000")
            print("  API documentation: http://localhost:8000/docs")
            print("  Press Ctrl+C to stop when finished")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"✗ API server failed to start: {stderr}")
            return None
    except Exception as e:
        print(f"✗ Failed to start API server: {e}")
        return None

def start_dashboard(dashboard_type='integrated'):
    """Start the dashboard"""
    print(f"\n== Starting {dashboard_type.capitalize()} Dashboard ==")
    
    if dashboard_type == 'integrated':
        script_path = os.path.join(PROJECT_ROOT, "integrated_dashboard.py")
    elif dashboard_type == 'realtime':
        script_path = os.path.join(PROJECT_ROOT, "real_time_monitor.py")
    else:
        script_path = os.path.join(PROJECT_ROOT, "app.py")
    
    try:
        # Start process in background
        process = subprocess.Popen([sys.executable, script_path],
                                  cwd=PROJECT_ROOT,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✓ {dashboard_type.capitalize()} dashboard started successfully")
            print("  URL: http://localhost:8050")
            print("  Press Ctrl+C to stop when finished")
            
            # Open web browser
            webbrowser.open('http://localhost:8050')
            
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"✗ Dashboard failed to start: {stderr}")
            return None
    except Exception as e:
        print(f"✗ Failed to start dashboard: {e}")
        return None

def start_simulation():
    """Start the hardware simulation"""
    print("\n== Starting Hardware Simulation ==")
    script_path = os.path.join(PROJECT_ROOT, "raspberry_pi_seu_detector.py")
    
    try:
        # Start process with simulation flag
        process = subprocess.Popen([sys.executable, script_path, "-s"],
                                  cwd=PROJECT_ROOT,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        # Wait a bit to see if it crashes immediately
        time.sleep(1)
        
        # Check if process is still running
        if process.poll() is None:
            print("✓ Hardware simulation started successfully")
            print("  Data collection started (simulation mode)")
            print("  Press Ctrl+C to stop when finished")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"✗ Simulation failed to start: {stderr}")
            return None
    except Exception as e:
        print(f"✗ Failed to start simulation: {e}")
        return None

def run_test_suite():
    """Run test suite with coverage report"""
    print("\n== Running Test Suite ==")
    
    script_path = os.path.join(PROJECT_ROOT, "run_tests.py")
    
    try:
        subprocess.run([sys.executable, script_path, "--coverage", "-v"],
                      cwd=PROJECT_ROOT, check=True)
        print("✓ Tests completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Tests failed with exit code {e.returncode}")
        return False

def interactive_menu():
    """Display an interactive menu for starting components"""
    print("\n=================================")
    print("  SEU Detector - Control Panel")
    print("=================================\n")
    
    print("What would you like to do?\n")
    print("1. Generate synthetic test data")
    print("2. Run ML pipeline")
    print("3. Train anomaly detection model")
    print("4. Start API server")
    print("5. Start integrated dashboard")
    print("6. Start real-time monitor")
    print("7. Start hardware simulation")
    print("8. Start complete system (API + Dashboard + Simulation)")
    print("9. Run test suite")
    print("10. Exit\n")
    
    choice = input("Enter your choice (1-10): ")
    
    processes = []
    
    try:
        if choice == '1':
            generate_synthetic_data()
        elif choice == '2':
            run_ml_pipeline()
        elif choice == '3':
            train_anomaly_detection()
        elif choice == '4':
            process = start_api_server()
            if process:
                processes.append(process)
                input("Press Enter to stop the server...")
        elif choice == '5':
            process = start_dashboard('integrated')
            if process:
                processes.append(process)
                input("Press Enter to stop the dashboard...")
        elif choice == '6':
            process = start_dashboard('realtime')
            if process:
                processes.append(process)
                input("Press Enter to stop the monitor...")
        elif choice == '7':
            process = start_simulation()
            if process:
                processes.append(process)
                input("Press Enter to stop the simulation...")
        elif choice == '8':
            # Start API server
            api_process = start_api_server()
            if api_process:
                processes.append(api_process)
                
                # Start simulation
                sim_process = start_simulation()
                if sim_process:
                    processes.append(sim_process)
                
                # Start dashboard
                dash_process = start_dashboard('integrated')
                if dash_process:
                    processes.append(dash_process)
                    
                    input("Press Enter to stop all services...")
        elif choice == '9':
            run_test_suite()
        elif choice == '10':
            print("Exiting...")
        else:
            print("Invalid choice, please try again")
            return interactive_menu()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up processes
        for process in processes:
            if process and process.poll() is None:
                process.terminate()
                print(f"Stopped process {process.pid}")
        
        print("All processes stopped")

def main():
    parser = argparse.ArgumentParser(description='SEU Detector Launcher')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--run-ml', action='store_true', help='Run ML pipeline')
    parser.add_argument('--train-anomaly', action='store_true', help='Train anomaly detection model')
    parser.add_argument('--api', action='store_true', help='Start API server')
    parser.add_argument('--dashboard', choices=['simple', 'realtime', 'integrated'], 
                      help='Start a dashboard (simple, realtime, or integrated)')
    parser.add_argument('--simulate', action='store_true', help='Start hardware simulation')
    parser.add_argument('--test', action='store_true', help='Run test suite with coverage report')
    parser.add_argument('--all', action='store_true', help='Start complete system')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    processes = []
    
    try:        # If no arguments, show interactive menu
        if not any(vars(args).values()):
            interactive_menu()
            return
        
        # Otherwise process command line arguments
        if args.test:
            run_test_suite()
            return
            
        if args.generate_data:
            generate_synthetic_data()
        
        if args.run_ml:
            run_ml_pipeline()
        
        if args.train_anomaly:
            train_anomaly_detection()
        
        if args.api or args.all:
            api_process = start_api_server()
            if api_process:
                processes.append(api_process)
        
        if args.simulate or args.all:
            sim_process = start_simulation()
            if sim_process:
                processes.append(sim_process)
        
        if args.dashboard or args.all:
            dashboard_type = args.dashboard if args.dashboard else 'integrated'
            dash_process = start_dashboard(dashboard_type)
            if dash_process:
                processes.append(dash_process)
        
        # If any background processes were started, wait for user to stop them
        if processes:
            print("\nPress Ctrl+C to stop all services...")
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up processes
        for process in processes:
            if process and process.poll() is None:
                process.terminate()
                print(f"Stopped process {process.pid}")
        
        print("All processes stopped")

if __name__ == "__main__":
    main()
