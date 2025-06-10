# SRAM SEU Detector for APEX

A comprehensive system for detecting, analyzing, and predicting Single Event Upsets (SEUs) in SRAM memory caused by cosmic radiation.

## Overview

This project consists of multiple components:
- A Raspberry Pi Zero 2 W with SRAM chips, environmental sensors, and GPS
- Two Pico W microcontrollers for distributed data collection
- Machine learning pipeline for analyzing SEU events and anomaly detection
- Real-time monitoring dashboard
- Historical data visualization
- REST API for integration with external systems

The system collects data on Single-Event Upsets (SEUs) in SRAM memory, such as bit flips, from ground level up to 30 km in altitude, along with environmental parameters (pressure, altitude, temperature) and location. This data is used to train machine learning models that can predict SEU events in real time.

<picture>
 <img alt="Main Component" src="images/IMG_5444.png" width="400">
</picture>

<picture>
  <img alt="SRAM" src="images/IMG_5447.png" width="400">
</picture>

## Components

### Hardware
- **Raspberry Pi Zero 2 W**: Main controller
- **Pico W Microcontrollers**: Distributed data collection
- **SRAM Chips**: Memory tested for SEU events
- **BMP280 Sensor**: Temperature and pressure measurements
- **GPS Module**: Location tracking
- **Cosmic Ray Detector**: Optional radiation counter

### Software
- **Data Collection**: Python scripts for hardware interaction and data collection
- **ML Pipeline**: Machine learning models for SEU prediction and anomaly detection
- **Real-time Monitor**: Dashboard for live monitoring of SEU events
- **API Server**: RESTful API for data access and integration
- **Integrated Dashboard**: Comprehensive visualization of real-time and historical data

## Installation

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

Quick start:

```bash
# Clone repository
git clone https://github.com/yourusername/seu-detector.git
cd seu-detector

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (if no real data available)
cd ml_pipeline
python SEU-Synthetic-Data-Creator.py

# Run ML pipeline
python SEU-ML-Pipeline.py

# Start dashboard
cd ..
python integrated_dashboard.py
```

## CI/CD Pipeline

This project includes automated testing and deployment through a CI/CD pipeline implemented with GitHub Actions. 
The pipeline runs tests, checks code quality, and builds Docker containers for deployment.

See [CI_CD.md](CI_CD.md) for detailed information about the CI/CD setup.

## Results

The system provides insights into SEU occurrences based on altitude, temperature, and cosmic radiation intensity. The machine learning models can predict SEU events with high accuracy based on environmental factors.

<picture>
  <img alt="Synthetic Data Analysis" src="images/seu_synthetic_analysis_results.png" width="604">
</picture>

## Team

Members:
Yasashvee Karthi, Alexander Jameson, Om Ghosh, Atrey Iyer
