import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

def generate_realistic_seu_data(duration_hours=6, samples_per_minute=4):
    """
    Generate realistic synthetic SEU data for a high-altitude balloon flight
    
    Parameters:
    - duration_hours: Total flight duration
    - samples_per_minute: Data collection frequency
    """
    
    print("Generating Synthetic SRAM SEU Data")
    print("=" * 50)
    
    # Calculate total samples
    total_samples = duration_hours * 60 * samples_per_minute
    
    # Time array (in seconds from launch)
    time_seconds = np.linspace(0, duration_hours * 3600, total_samples)
    time_delta = np.diff(time_seconds, prepend=0)
    time_delta[0] = time_delta[1]  # Fix first value
    
    # Generate realistic altitude profile for balloon flight
    altitude = generate_altitude_profile(time_seconds, duration_hours)
    
    # Generate temperature based on altitude (gets colder with height)
    temperature = generate_temperature_profile(altitude)
    
    # Generate GPS coordinates (realistic balloon drift)
    latitude, longitude = generate_gps_coordinates(time_seconds)
    
    # Generate cosmic ray intensity based on altitude
    cosmic_intensity = calculate_cosmic_intensity(altitude)
    
    # Generate realistic SEU events
    bit_flips_count = generate_seu_events(altitude, cosmic_intensity, temperature, time_seconds)
    
    # Generate max run length (consecutive bit flips)
    max_run_length = generate_run_lengths(bit_flips_count)
    
    # Generate bit error distribution across SRAM regions
    sram_region_errors = generate_sram_region_errors(bit_flips_count)
    
    # Add some realistic noise and occasional bursts
    bit_flips_count = add_cosmic_ray_bursts(bit_flips_count, time_seconds)
    
    # Create DataFrame
    data = {
        'timestamp': [datetime.now() + timedelta(seconds=int(t)) for t in time_seconds],
        'time_seconds': time_seconds,
        'time_delta': time_delta,
        'altitude': altitude,
        'temperature': temperature,
        'latitude': latitude,
        'longitude': longitude,
        'bit_flips_count': bit_flips_count.astype(int),
        'max_run_length': max_run_length.astype(int),
        'sram_region_0_errors': sram_region_errors[:, 0].astype(int),
        'sram_region_1_errors': sram_region_errors[:, 1].astype(int),
        'sram_region_2_errors': sram_region_errors[:, 2].astype(int),
        'sram_region_3_errors': sram_region_errors[:, 3].astype(int),
        'cosmic_intensity': cosmic_intensity
    }
    
    df = pd.DataFrame(data)
    
    # Add some measurement noise
    df = add_measurement_noise(df)
    
    return df

def generate_altitude_profile(time_seconds, duration_hours):
    """Generate realistic balloon altitude profile"""
    max_altitude = 35000  # ~35km typical balloon altitude
    ascent_time = duration_hours * 0.4 * 3600  # 40% of flight is ascent
    float_time = duration_hours * 0.4 * 3600   # 40% at float altitude
    descent_time = duration_hours * 0.2 * 3600 # 20% descent
    
    altitude = np.zeros_like(time_seconds)
    
    for i, t in enumerate(time_seconds):
        if t <= ascent_time:
            # Exponential-like ascent (slower at first, then faster)
            progress = t / ascent_time
            altitude[i] = max_altitude * (progress + 0.3 * progress**2)
        elif t <= ascent_time + float_time:
            # Float phase with small oscillations
            oscillation = 500 * np.sin(2 * np.pi * (t - ascent_time) / 1800)  # 30-min period
            altitude[i] = max_altitude + oscillation + np.random.normal(0, 100)
        else:
            # Descent phase
            descent_progress = (t - ascent_time - float_time) / descent_time
            altitude[i] = max_altitude * (1 - descent_progress**1.5)
    
    # Ensure no negative altitudes
    altitude = np.maximum(altitude, 0)
    
    return altitude

def generate_temperature_profile(altitude):
    """Generate temperature based on altitude (standard atmosphere model)"""
    # Standard atmosphere approximation
    sea_level_temp = 15  # °C
    lapse_rate = -6.5e-3  # °C/m up to tropopause
    tropopause_alt = 11000  # m
    tropopause_temp = -56.5  # °C
    
    temperature = np.zeros_like(altitude)
    
    for i, alt in enumerate(altitude):
        if alt <= tropopause_alt:
            temperature[i] = sea_level_temp + lapse_rate * alt
        else:
            # Stratosphere (constant temperature with slight warming)
            temperature[i] = tropopause_temp + 0.001 * (alt - tropopause_alt)
    
    # Add some measurement noise
    temperature += np.random.normal(0, 2, len(temperature))
    
    return temperature

def generate_gps_coordinates(time_seconds):
    """Generate realistic GPS coordinates for balloon drift"""
    # Starting position (example: over Kansas)
    start_lat = 39.0
    start_lon = -98.0
    
    # Simulate wind drift (realistic balloon movement)
    wind_speed_ms = 15  # m/s average wind
    wind_direction = np.pi / 4  # NE direction
    
    # Convert to lat/lon change per second
    lat_change_per_sec = (wind_speed_ms * np.cos(wind_direction)) / 111111  # ~111km per degree
    lon_change_per_sec = (wind_speed_ms * np.sin(wind_direction)) / (111111 * np.cos(np.radians(start_lat)))
    
    latitude = start_lat + lat_change_per_sec * time_seconds
    longitude = start_lon + lon_change_per_sec * time_seconds
    
    # Add some random variation for realistic GPS drift
    latitude += np.random.normal(0, 0.001, len(latitude))
    longitude += np.random.normal(0, 0.001, len(longitude))
    
    return latitude, longitude

def calculate_cosmic_intensity(altitude):
    """Calculate cosmic ray intensity based on altitude"""
    # Cosmic ray intensity increases exponentially with altitude
    sea_level_intensity = 1.0
    scale_height = 4500  # meters (approximation)
    
    intensity = sea_level_intensity * np.exp(altitude / scale_height)
    
    return intensity

def generate_seu_events(altitude, cosmic_intensity, temperature, time_seconds):
    """Generate realistic SEU events based on environmental conditions"""
    # Base SEU rate (events per second at sea level)
    base_rate = 1e-6  # Very low at sea level
    
    # SEU rate increases with cosmic intensity
    seu_rate = base_rate * cosmic_intensity
    
    # Temperature effect (silicon becomes more susceptible when very cold)
    temp_factor = np.where(temperature < -40, 1.2, 1.0)  # 20% increase when very cold
    seu_rate *= temp_factor
    
    # Generate Poisson-distributed events
    bit_flips = np.random.poisson(seu_rate * 15, len(altitude))  # 15-sec intervals
    
    # Add altitude-dependent scaling
    altitude_factor = 1 + (altitude / 10000) ** 1.5  # Stronger effect at high altitudes
    bit_flips = (bit_flips * altitude_factor).astype(float)
    
    return bit_flips

def generate_run_lengths(bit_flips_count):
    """Generate maximum run length for consecutive bit flips"""
    max_run_length = np.zeros_like(bit_flips_count)
    
    for i, flips in enumerate(bit_flips_count):
        if flips > 0:
            # Most single flips, some clusters
            if flips == 1:
                max_run_length[i] = 1
            elif flips <= 5:
                max_run_length[i] = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            else:
                # For high flip counts, longer runs are possible
                max_run_length[i] = min(flips, np.random.poisson(2) + 1)
        else:
            max_run_length[i] = 0
    
    return max_run_length

def generate_sram_region_errors(bit_flips_count):
    """Distribute bit flips across different SRAM regions"""
    num_regions = 4
    region_errors = np.zeros((len(bit_flips_count), num_regions))
    
    for i, total_flips in enumerate(bit_flips_count):
        if total_flips > 0:
            # Randomly distribute flips across regions
            region_distribution = np.random.multinomial(int(total_flips), 
                                                      [0.25, 0.25, 0.25, 0.25])
            region_errors[i] = region_distribution
    
    return region_errors

def add_cosmic_ray_bursts(bit_flips, time_seconds):
    """Add occasional cosmic ray shower bursts"""
    # Add ~5-10 burst events during the flight
    num_bursts = np.random.randint(5, 11)
    
    for _ in range(num_bursts):
        # Random time for burst
        burst_time_idx = np.random.randint(0, len(time_seconds))
        
        # Burst intensity (10-50x normal rate)
        burst_intensity = np.random.randint(10, 51)
        
        # Burst duration (30 seconds to 5 minutes)
        burst_duration_samples = np.random.randint(8, 80)  # 4 samples/minute
        
        # Apply burst
        end_idx = min(burst_time_idx + burst_duration_samples, len(bit_flips))
        bit_flips[burst_time_idx:end_idx] *= burst_intensity
    
    return bit_flips

def add_measurement_noise(df):
    """Add realistic measurement noise and occasional missing data"""
    # Temperature sensor noise
    df['temperature'] += np.random.normal(0, 0.5, len(df))
    
    # Altitude pressure sensor noise
    df['altitude'] += np.random.normal(0, 10, len(df))
    df['altitude'] = np.maximum(df['altitude'], 0)  # No negative altitudes
    
    # Occasionally miss some bit flip detections (false negatives)
    false_negative_mask = np.random.random(len(df)) < 0.02  # 2% false negative rate
    df.loc[false_negative_mask, 'bit_flips_count'] = np.maximum(
        df.loc[false_negative_mask, 'bit_flips_count'] - 1, 0)
    
    # Occasionally add false positives (EMI, temperature effects)
    false_positive_mask = np.random.random(len(df)) < 0.01  # 1% false positive rate
    df.loc[false_positive_mask, 'bit_flips_count'] += np.random.randint(1, 4, 
                                                                        sum(false_positive_mask))
    
    return df

def plot_synthetic_data(df):
    """Plot the synthetic data to verify it looks realistic"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Altitude profile
    axes[0, 0].plot(df['time_seconds']/3600, df['altitude']/1000)
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Altitude (km)')
    axes[0, 0].set_title('Balloon Flight Profile')
    axes[0, 0].grid(True)
    
    # Temperature vs altitude
    axes[0, 1].scatter(df['altitude']/1000, df['temperature'], alpha=0.6, s=1)
    axes[0, 1].set_xlabel('Altitude (km)')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_title('Temperature vs Altitude')
    axes[0, 1].grid(True)
    
    # SEU events over time
    axes[1, 0].plot(df['time_seconds']/3600, df['bit_flips_count'])
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Bit Flips Count')
    axes[1, 0].set_title('SEU Events During Flight')
    axes[1, 0].grid(True)
    
    # SEU vs altitude
    axes[1, 1].scatter(df['altitude']/1000, df['bit_flips_count'], alpha=0.6)
    axes[1, 1].set_xlabel('Altitude (km)')
    axes[1, 1].set_ylabel('Bit Flips Count')
    axes[1, 1].set_title('SEU Events vs Altitude')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Generate and save synthetic SEU data"""
    print("Generating synthetic SEU data for high-altitude balloon experiment...")
    
    # Generate data
    df = generate_realistic_seu_data(duration_hours=6, samples_per_minute=4)
    
    # Display basic statistics
    print(f"\nGenerated {len(df)} data points over {df['time_seconds'].max()/3600:.1f} hours")
    print(f"Altitude range: {df['altitude'].min():.0f} - {df['altitude'].max():.0f} m")
    print(f"Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f} °C")
    print(f"Total SEU events: {df['bit_flips_count'].sum()}")
    print(f"Peak SEU rate: {df['bit_flips_count'].max()} events/sample")
    
    # Save to CSV
    df.to_csv('seu_synthetic_data.csv', index=False)
    print(f"\nSynthetic data saved to 'seu_synthetic_data.csv'")
    
    # Plot the data
    plot_synthetic_data(df)
    
    return df

if __name__ == "__main__":
    synthetic_data = main()
