import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import threading
import time
import os
from datetime import datetime
from collections import deque

# Import serial if available
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    print("Warning: Serial module not available - using simulation mode")
    HAS_SERIAL = False

# Configuration
UPDATE_INTERVAL = 1  # seconds
MAX_POINTS = 100     # Maximum number of points to display
DATA_FILE = "real_time_seu_data.csv"

# Initialize data storage
timestamps = deque(maxlen=MAX_POINTS)
seu_counts = deque(maxlen=MAX_POINTS)
altitudes = deque(maxlen=MAX_POINTS)
temperatures = deque(maxlen=MAX_POINTS)
pressures = deque(maxlen=MAX_POINTS)

# Serial connection parameters
SERIAL_PORT = 'COM3'  # Change to your Pi's serial port
BAUD_RATE = 9600

# Initialize serial connection (commented out until hardware is ready)
serial_connected = False
ser = None

# Only attempt serial connection if module is available
if HAS_SERIAL:
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        serial_connected = True
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Warning: Could not connect to serial port: {e}")
        print("Running in simulation mode")
else:
    print("Serial module not available - running in simulation mode")

# Function to read data from serial port or simulate data
def read_data():
    if serial_connected and ser:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                return json.loads(line)
        except Exception as e:
            print(f"Serial read error: {e}")
    
    # Simulation mode - generate fake data
    import numpy as np
    
    # Generate simulated data
    alt = 10000 + np.random.normal(0, 500)
    temp = -20 + np.random.normal(0, 2)
    press = 300 + np.random.normal(0, 5)
    cosmic_intensity = 15 + np.random.normal(0, 1)
    seu_count = int(max(0, np.random.poisson(cosmic_intensity / 5)))
    
    return {
        "timestamp": datetime.now().isoformat(),
        "altitude": alt,
        "temperature": temp,
        "pressure": press,
        "seu_count": seu_count
    }

# Function to save data to CSV
def save_data_to_csv(data):
    file_exists = os.path.isfile(DATA_FILE)
    with open(DATA_FILE, 'a') as f:
        if not file_exists:
            # Write header
            f.write("timestamp,altitude,temperature,pressure,seu_count\n")
        
        # Write data
        f.write(f"{data['timestamp']},{data['altitude']},{data['temperature']},{data['pressure']},{data['seu_count']}\n")

# Data collection thread
def data_collection_thread():
    while True:
        data = read_data()
        
        # Add to deques
        timestamps.append(data['timestamp'])
        seu_counts.append(data['seu_count'])
        altitudes.append(data['altitude'])
        temperatures.append(data['temperature'])
        pressures.append(data['pressure'])
        
        # Save to CSV
        save_data_to_csv(data)
        
        time.sleep(UPDATE_INTERVAL)

# Start data collection in a separate thread
data_thread = threading.Thread(target=data_collection_thread, daemon=True)
data_thread.start()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Real-Time SEU Monitor"

# App layout
app.layout = html.Div(style={'padding': '20px', 'font-family': 'Arial'}, children=[
    html.H1("🔬 Real-Time SRAM SEU Monitor", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H3("Current Status", style={'textAlign': 'center'}),
            html.Div(id='status-display', style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'justifyContent': 'space-around',
                'marginBottom': '20px'
            })
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Div([
                html.H3("SEU Events (Live)", style={'textAlign': 'center'}),
                dcc.Graph(id='live-seu-graph')
            ], style={'width': '50%'}),
            
            html.Div([
                html.H3("Altitude & Temperature (Live)", style={'textAlign': 'center'}),
                dcc.Graph(id='live-alt-temp-graph')
            ], style={'width': '50%'})
        ], style={'display': 'flex'}),
        
        html.Div([
            html.H3("SEU Events vs. Altitude (Live)", style={'textAlign': 'center'}),
            dcc.Graph(id='live-seu-altitude-correlation')
        ]),
        
        dcc.Interval(
            id='interval-component',
            interval=1000,  # in milliseconds
            n_intervals=0
        )
    ])
])

# Callback for status display
@app.callback(
    Output('status-display', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_status_display(n):
    if not timestamps:
        return [html.Div("No data available")]
    
    # Get latest values
    latest_seu = seu_counts[-1] if seu_counts else 0
    latest_alt = altitudes[-1] if altitudes else 0
    latest_temp = temperatures[-1] if temperatures else 0
    latest_press = pressures[-1] if pressures else 0
    
    # Alert level based on SEU count
    alert_level = "Low"
    alert_color = "green"
    
    if latest_seu >= 10:
        alert_level = "High"
        alert_color = "red"
    elif latest_seu >= 3:
        alert_level = "Medium"
        alert_color = "orange"
    
    status_boxes = [
        html.Div([
            html.H4("SEU Rate"),
            html.H2(f"{latest_seu}", style={'color': alert_color}),
            html.P(f"Alert Level: {alert_level}", style={'color': alert_color})
        ], style={'border': f'2px solid {alert_color}', 'borderRadius': '5px', 'padding': '10px', 'width': '200px', 'textAlign': 'center'}),
        
        html.Div([
            html.H4("Altitude"),
            html.H2(f"{latest_alt:.0f} m"),
        ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'width': '200px', 'textAlign': 'center'}),
        
        html.Div([
            html.H4("Temperature"),
            html.H2(f"{latest_temp:.1f}°C"),
        ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'width': '200px', 'textAlign': 'center'}),
        
        html.Div([
            html.H4("Pressure"),
            html.H2(f"{latest_press:.1f} hPa"),
        ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'width': '200px', 'textAlign': 'center'})
    ]
    
    return status_boxes

# Callback for SEU graph
@app.callback(
    Output('live-seu-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_seu_graph(n):
    if not timestamps:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(timestamps), 
        y=list(seu_counts),
        mode='lines+markers',
        name='SEU Count',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="SEU Count",
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=True, gridcolor='#eee')
    )
    
    return fig

# Callback for altitude & temperature graph
@app.callback(
    Output('live-alt-temp-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_alt_temp_graph(n):
    if not timestamps:
        return go.Figure()
    
    fig = go.Figure()
    
    # Add altitude trace
    fig.add_trace(go.Scatter(
        x=list(timestamps), 
        y=list(altitudes),
        mode='lines',
        name='Altitude (m)',
        line=dict(color='blue', width=2),
        yaxis='y'
    ))
      # Add temperature trace
    fig.add_trace(go.Scatter(
        x=list(timestamps), 
        y=list(temperatures),
        mode='lines',
        name='Temperature (°C)',
        line=dict(color='orange', width=2),
        yaxis='y2'
    ))
    
    # Layout with secondary y-axis
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Altitude (m)",
        yaxis2=dict(
            title="Temperature (°C)",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        margin=dict(l=20, r=50, t=30, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=True, gridcolor='#eee'),
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

# Callback for SEU vs altitude scatter plot
@app.callback(
    Output('live-seu-altitude-correlation', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_seu_alt_correlation(n):
    if not timestamps or len(altitudes) < 2:
        return go.Figure()
    
    fig = go.Figure()
    
    # Create scatter plot
    fig.add_trace(go.Scatter(
        x=list(altitudes), 
        y=list(seu_counts),
        mode='markers',
        marker=dict(
            size=10,
            color=list(temperatures),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Temperature (°C)")
        ),
        text=[f"Time: {t}<br>SEUs: {s}<br>Alt: {a:.0f}m<br>Temp: {temp:.1f}°C" 
              for t, s, a, temp in zip(timestamps, seu_counts, altitudes, temperatures)],
        hoverinfo='text'
    ))
    
    fig.update_layout(
        xaxis_title="Altitude (m)",
        yaxis_title="SEU Count",
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=True, gridcolor='#eee')
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    print("=" * 50)
    print("Starting Real-Time SEU Monitor")
    print("Dashboard URL: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    app.run_server(debug=False)
