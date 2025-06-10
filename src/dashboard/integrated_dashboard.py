import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Import requests if available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    print("Warning: Requests module not available - API access will be simulated")
    HAS_REQUESTS = False
import time
import threading
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_URL = "http://localhost:8000"  # URL for the API server
UPDATE_INTERVAL = 2000  # 2 seconds
MAX_POINTS = 500  # Maximum points to keep in memory
HISTORICAL_DATA_FILE = "ml_pipeline/seu_analysis_results.csv"

# Initialize app
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="SEU Detector Dashboard"
)

# Data storage
latest_data = {}
historical_df = None
realtime_data = []

# Load historical data
def load_historical_data():
    global historical_df
    try:
        if os.path.exists(HISTORICAL_DATA_FILE):
            df = pd.read_csv(HISTORICAL_DATA_FILE)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            historical_df = df
            print(f"Loaded {len(df)} historical data points")
        else:
            # Fall back to synthetic data
            if os.path.exists("ml_pipeline/seu_synthetic_data.csv"):
                df = pd.read_csv("ml_pipeline/seu_synthetic_data.csv")
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                historical_df = df
                print(f"Loaded {len(df)} synthetic data points")
    except Exception as e:
        print(f"Error loading historical data: {e}")
        historical_df = pd.DataFrame()

# Fetch data from API
def fetch_latest_data():
    global latest_data, realtime_data
    
    if HAS_REQUESTS:
        try:
            response = requests.get(f"{API_URL}/data/latest")
            if response.status_code == 200:
                data = response.json()
                latest_data = data
                
                # Add timestamp if missing
                if 'timestamp' not in data:
                    data['timestamp'] = datetime.now().isoformat()
                    
                # Add to realtime data list
                realtime_data.append(data)
                
                # Keep only the last MAX_POINTS
                if len(realtime_data) > MAX_POINTS:
                    realtime_data = realtime_data[-MAX_POINTS:]
                    
                return data
        except Exception as e:
            print(f"API error: {e}")
            # Fall through to simulation
    
    # Simulate data if API fails or requests not available
    sim_data = {
        'timestamp': datetime.now().isoformat(),
        'altitude': 15000 + np.random.normal(0, 500),
        'temperature': -20 + np.random.normal(0, 2),
        'pressure': 300 + np.random.normal(0, 5),
        'bit_flips_count': int(max(0, np.random.poisson(3))),
        'cosmic_intensity': 15 + np.random.normal(0, 1)
    }
    
    latest_data = sim_data
    realtime_data.append(sim_data)
    
    # Keep only the last MAX_POINTS
    if len(realtime_data) > MAX_POINTS:
        realtime_data = realtime_data[-MAX_POINTS:]
    
    return sim_data

# Load historical data at startup
load_historical_data()

# Define app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("SRAM SEU Detector Dashboard", style={'textAlign': 'center'}),
        html.P("Monitoring Single Event Upsets in SRAM Memory", style={'textAlign': 'center'})
    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa'}),
    
    # Tabs
    dcc.Tabs([
        # Real-time data tab
        dcc.Tab(label="Real-time Monitor", children=[
            html.Div([
                # Status cards row
                html.Div([
                    html.Div([
                        html.H4("Current Status"),
                        html.Div(id="status-cards", style={
                            'display': 'flex',
                            'flexWrap': 'wrap',
                            'justifyContent': 'space-between'
                        })
                    ], style={'margin': '20px 0'})
                ]),
                
                # Main charts row
                html.Div([
                    # SEU Events chart
                    html.Div([
                        html.H4("SEU Events (Real-time)"),
                        dcc.Graph(id="seu-events-chart")
                    ], style={'width': '50%', 'padding': '10px', 'boxSizing': 'border-box'}),
                    
                    # Altitude & Temperature chart
                    html.Div([
                        html.H4("Flight Profile (Real-time)"),
                        dcc.Graph(id="altitude-temp-chart")
                    ], style={'width': '50%', 'padding': '10px', 'boxSizing': 'border-box'})
                ], style={'display': 'flex', 'flexWrap': 'wrap'}),
                
                # Anomaly detection row
                html.Div([
                    html.H4("SEU Events vs Flight Parameters"),
                    dcc.Graph(id="anomaly-detection-chart")
                ], style={'margin': '20px 0'}),
                
                # Interval component for updates
                dcc.Interval(
                    id="interval-component",
                    interval=UPDATE_INTERVAL,
                    n_intervals=0,
                ),
            ], style={'padding': '20px'})
        ]),
        
        # Historical data analysis tab
        dcc.Tab(label="Historical Analysis", children=[
            html.Div([
                # Date range selector
                html.Div([
                    html.H4("Select Date Range"),                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d'),
                        max_date_allowed=datetime.now().strftime('%Y-%m-%d'),
                        initial_visible_month=datetime.now().strftime('%Y-%m-%d')
                    ),
                    html.Button('Apply', id='apply-date-button', n_clicks=0)
                ], style={'margin': '20px 0'}),
                
                # Summary statistics
                html.Div([
                    html.H4("Summary Statistics"),
                    html.Div(id="summary-stats", style={
                        'display': 'flex',
                        'flexWrap': 'wrap',
                        'justifyContent': 'space-between'
                    })
                ], style={'margin': '20px 0'}),
                
                # Historical charts
                html.Div([
                    # First row
                    html.Div([
                        # SEU Heatmap
                        html.Div([
                            html.H4("SEU Events Heatmap"),
                            dcc.Graph(id="seu-heatmap")
                        ], style={'width': '50%', 'padding': '10px', 'boxSizing': 'border-box'}),
                        
                        # Correlation Matrix
                        html.Div([
                            html.H4("Parameter Correlation"),
                            dcc.Graph(id="correlation-matrix")
                        ], style={'width': '50%', 'padding': '10px', 'boxSizing': 'border-box'})
                    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
                    
                    # Second row
                    html.Div([
                        # SEU Altitude Relationship
                        html.Div([
                            html.H4("SEU vs Altitude Relationship"),
                            dcc.Graph(id="seu-altitude-chart")
                        ], style={'width': '50%', 'padding': '10px', 'boxSizing': 'border-box'}),
                        
                        # Temperature Relationship
                        html.Div([
                            html.H4("SEU vs Temperature Relationship"),
                            dcc.Graph(id="seu-temp-chart")
                        ], style={'width': '50%', 'padding': '10px', 'boxSizing': 'border-box'})
                    ], style={'display': 'flex', 'flexWrap': 'wrap'})
                ])
            ], style={'padding': '20px'})
        ]),
        
        # Map visualization tab
        dcc.Tab(label="Geographical Map", children=[
            html.Div([
                html.H4("SEU Events Geographical Distribution"),
                dcc.Graph(id="geo-map", style={'height': '800px'}),
                html.Div([
                    html.P("Map shows the geographical distribution of SEU events during flight."),
                    html.P("Color intensity indicates SEU event frequency, size indicates severity.")
                ])
            ], style={'padding': '20px'})
        ])
    ])
])

# Callback for updating status cards
@app.callback(
    Output("status-cards", "children"),
    Input("interval-component", "n_intervals")
)
def update_status_cards(n):
    # Fetch latest data
    data = fetch_latest_data()
    
    if not data:
        return [html.Div("No data available - API connection failed")]
    
    # Define status cards
    cards = []
    
    # SEU Rate card
    seu_count = data.get('bit_flips_count', 0)
    alert_level = "Low"
    alert_color = "#28a745"  # green
    
    if seu_count > 10:
        alert_level = "High"
        alert_color = "#dc3545"  # red
    elif seu_count > 3:
        alert_level = "Medium"
        alert_color = "#ffc107"  # yellow
    
    cards.append(html.Div([
        html.H5("SEU Rate"),
        html.H3(f"{seu_count}", style={'color': alert_color}),
        html.P(f"Alert Level: {alert_level}", style={'color': alert_color})
    ], style={
        'border': f'1px solid {alert_color}',
        'borderLeft': f'5px solid {alert_color}',
        'borderRadius': '5px',
        'padding': '15px',
        'margin': '5px',
        'width': '200px',
        'backgroundColor': '#f8f9fa'
    }))
    
    # Altitude card
    altitude = data.get('altitude', 0)
    cards.append(html.Div([
        html.H5("Altitude"),
        html.H3(f"{altitude:.0f} m"),
        html.P("Above sea level")
    ], style={
        'border': '1px solid #007bff',
        'borderLeft': '5px solid #007bff',
        'borderRadius': '5px',
        'padding': '15px',
        'margin': '5px',
        'width': '200px',
        'backgroundColor': '#f8f9fa'
    }))
    
    # Temperature card
    temperature = data.get('temperature', 0)
    cards.append(html.Div([
        html.H5("Temperature"),
        html.H3(f"{temperature:.1f}°C"),
        html.P("Current reading")
    ], style={
        'border': '1px solid #6f42c1',
        'borderLeft': '5px solid #6f42c1',
        'borderRadius': '5px',
        'padding': '15px',
        'margin': '5px',
        'width': '200px',
        'backgroundColor': '#f8f9fa'
    }))
    
    # Cosmic intensity card
    cosmic = data.get('cosmic_intensity', altitude/10000 if altitude else 1)
    cards.append(html.Div([
        html.H5("Cosmic Intensity"),
        html.H3(f"{cosmic:.2f}"),
        html.P("Relative level")
    ], style={
        'border': '1px solid #fd7e14',
        'borderLeft': '5px solid #fd7e14',
        'borderRadius': '5px',
        'padding': '15px',
        'margin': '5px',
        'width': '200px',
        'backgroundColor': '#f8f9fa'
    }))
    
    return cards

# Callback for SEU events chart
@app.callback(
    Output("seu-events-chart", "figure"),
    Input("interval-component", "n_intervals")
)
def update_seu_events_chart(n):
    # Prepare data
    if not realtime_data:
        return go.Figure().update_layout(title="No data available")
    
    df = pd.DataFrame(realtime_data)
    
    # Ensure timestamp is in datetime format
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create figure
    fig = go.Figure()
    
    # Add SEU events line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['bit_flips_count'],
        mode='lines+markers',
        name='SEU Events',
        line=dict(color='red', width=2),
        marker=dict(size=7)
    ))
    
    # Add anomalies if available
    if 'is_anomaly' in df.columns and df['is_anomaly'].sum() > 0:
        anomaly_df = df[df['is_anomaly']]
        fig.add_trace(go.Scatter(
            x=anomaly_df['timestamp'],
            y=anomaly_df['bit_flips_count'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=12, symbol='star')
        ))
    
    # Layout
    fig.update_layout(
        title="SEU Events Over Time",
        xaxis_title="Time",
        yaxis_title="SEU Count",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Callback for altitude & temperature chart
@app.callback(
    Output("altitude-temp-chart", "figure"),
    Input("interval-component", "n_intervals")
)
def update_altitude_temp_chart(n):
    # Prepare data
    if not realtime_data:
        return go.Figure().update_layout(title="No data available")
    
    df = pd.DataFrame(realtime_data)
    
    # Ensure timestamp is in datetime format
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add altitude trace
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['altitude'],
        mode='lines',
        name='Altitude (m)',
        line=dict(color='blue', width=2)
    ))
    
    # Add temperature trace with secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['temperature'],
        mode='lines',
        name='Temperature (°C)',
        line=dict(color='orange', width=2),
        yaxis="y2"
    ))
    
    # Layout with secondary y-axis
    fig.update_layout(
        title="Flight Profile",
        xaxis_title="Time",
        yaxis_title="Altitude (m)",
        yaxis2=dict(
            title="Temperature (°C)",
            overlaying="y",
            side="right"
        ),
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Callback for anomaly detection chart
@app.callback(
    Output("anomaly-detection-chart", "figure"),
    Input("interval-component", "n_intervals")
)
def update_anomaly_detection_chart(n):
    # Prepare data
    if not realtime_data:
        return go.Figure().update_layout(title="No data available")
    
    df = pd.DataFrame(realtime_data)
    
    # Create scatter plot
    fig = go.Figure()
    
    marker_size = df['bit_flips_count'] + 5
    marker_size = np.clip(marker_size, 5, 20)
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=df['altitude'],
        y=df['bit_flips_count'],
        mode='markers',
        name='SEU Events',
        marker=dict(
            size=marker_size,
            color=df['temperature'],
            colorscale='Viridis',
            colorbar=dict(title="Temperature (°C)"),
            showscale=True
        ),
        text=[f"Time: {t}<br>SEUs: {s}<br>Alt: {a:.0f}m<br>Temp: {temp:.1f}°C" 
              for t, s, a, temp in zip(df['timestamp'], df['bit_flips_count'], 
                                      df['altitude'], df['temperature'])]
    ))
    
    # Layout
    fig.update_layout(
        title="SEU Events vs Flight Parameters",
        xaxis_title="Altitude (m)",
        yaxis_title="SEU Count",
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# Callback for updating summary statistics
@app.callback(
    Output("summary-stats", "children"),
    [Input("apply-date-button", "n_clicks")],
    [State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date")]
)
def update_summary_stats(n_clicks, start_date, end_date):
    global historical_df
    
    if historical_df is None or historical_df.empty:
        return [html.Div("No historical data available")]
    
    # Filter by date range if timestamps are available
    filtered_df = historical_df.copy()
    if 'timestamp' in filtered_df.columns:
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & 
                                     (filtered_df['timestamp'] <= end_date)]
    
    # Calculate statistics
    stats_cards = []
    
    # Total records
    stats_cards.append(html.Div([
        html.H5("Total Records"),
        html.H3(f"{len(filtered_df):,}"),
        html.P("Data points")
    ], style={
        'border': '1px solid #17a2b8',
        'borderLeft': '5px solid #17a2b8',
        'borderRadius': '5px',
        'padding': '15px',
        'margin': '5px',
        'width': '200px',
        'backgroundColor': '#f8f9fa'
    }))
    
    # Max altitude
    if 'altitude' in filtered_df.columns:
        max_alt = filtered_df['altitude'].max()
        stats_cards.append(html.Div([
            html.H5("Max Altitude"),
            html.H3(f"{max_alt:,.0f} m"),
            html.P("Highest point")
        ], style={
            'border': '1px solid #007bff',
            'borderLeft': '5px solid #007bff',
            'borderRadius': '5px',
            'padding': '15px',
            'margin': '5px',
            'width': '200px',
            'backgroundColor': '#f8f9fa'
        }))
    
    # Total SEU events
    if 'bit_flips_count' in filtered_df.columns:
        total_seus = filtered_df['bit_flips_count'].sum()
        stats_cards.append(html.Div([
            html.H5("Total SEUs"),
            html.H3(f"{total_seus:,}"),
            html.P("Bit flips detected")
        ], style={
            'border': '1px solid #dc3545',
            'borderLeft': '5px solid #dc3545',
            'borderRadius': '5px',
            'padding': '15px',
            'margin': '5px',
            'width': '200px',
            'backgroundColor': '#f8f9fa'
        }))
    
    # Max SEU rate
    if 'bit_flips_count' in filtered_df.columns:
        max_seu = filtered_df['bit_flips_count'].max()
        stats_cards.append(html.Div([
            html.H5("Max SEU Rate"),
            html.H3(f"{max_seu}"),
            html.P("Highest count")
        ], style={
            'border': '1px solid #fd7e14',
            'borderLeft': '5px solid #fd7e14',
            'borderRadius': '5px',
            'padding': '15px',
            'margin': '5px',
            'width': '200px',
            'backgroundColor': '#f8f9fa'
        }))
    
    # Anomaly count
    if 'is_anomaly' in filtered_df.columns:
        anomaly_count = filtered_df['is_anomaly'].sum()
        stats_cards.append(html.Div([
            html.H5("Anomalies"),
            html.H3(f"{anomaly_count}"),
            html.P("Unusual events")
        ], style={
            'border': '1px solid #6f42c1',
            'borderLeft': '5px solid #6f42c1',
            'borderRadius': '5px',
            'padding': '15px',
            'margin': '5px',
            'width': '200px',
            'backgroundColor': '#f8f9fa'
        }))
    
    return stats_cards

# Callback for SEU heatmap
@app.callback(
    Output("seu-heatmap", "figure"),
    [Input("apply-date-button", "n_clicks")],
    [State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date")]
)
def update_seu_heatmap(n_clicks, start_date, end_date):
    global historical_df
    
    if historical_df is None or historical_df.empty:
        return go.Figure().update_layout(title="No historical data available")
    
    # Filter by date range
    filtered_df = historical_df.copy()
    if 'timestamp' in filtered_df.columns:
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & 
                                     (filtered_df['timestamp'] <= end_date)]
    
    # Check for required columns
    if 'altitude' not in filtered_df.columns or 'bit_flips_count' not in filtered_df.columns:
        return go.Figure().update_layout(title="Required columns not available")
      # Create altitude and temperature bins with simpler approach
    # Create bins as strings for easier plotting
    num_bins = 10
    altitude_min = filtered_df['altitude'].min()
    altitude_max = filtered_df['altitude'].max()
    altitude_step = (altitude_max - altitude_min) / num_bins
    
    temp_min = filtered_df['temperature'].min()
    temp_max = filtered_df['temperature'].max()
    temp_step = (temp_max - temp_min) / num_bins
    
    # Add bin labels
    filtered_df['altitude_bin'] = pd.cut(
        filtered_df['altitude'], 
        bins=num_bins,
        labels=[f"{int(altitude_min + i*altitude_step)}-{int(altitude_min + (i+1)*altitude_step)}" 
                for i in range(num_bins)]
    )
    
    filtered_df['temp_bin'] = pd.cut(
        filtered_df['temperature'], 
        bins=num_bins,
        labels=[f"{temp_min + i*temp_step:.1f}-{temp_min + (i+1)*temp_step:.1f}" 
                for i in range(num_bins)]
    )
    
    # Group data
    pivot_table = pd.pivot_table(
        filtered_df,
        values='bit_flips_count',
        index='altitude_bin',
        columns='temp_bin',
        aggfunc='mean',
        fill_value=0
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns.tolist(),
        y=pivot_table.index.tolist(),
        colorscale='Viridis',
        colorbar=dict(title="Avg SEU Count")
    ))
    
    # Update layout
    fig.update_layout(
        title="SEU Events by Altitude and Temperature",
        xaxis_title="Temperature Range (°C)",
        yaxis_title="Altitude Range (m)",
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# Callback for correlation matrix
@app.callback(
    Output("correlation-matrix", "figure"),
    [Input("apply-date-button", "n_clicks")],
    [State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date")]
)
def update_correlation_matrix(n_clicks, start_date, end_date):
    global historical_df
    
    if historical_df is None or historical_df.empty:
        return go.Figure().update_layout(title="No historical data available")
    
    # Filter by date range
    filtered_df = historical_df.copy()
    if 'timestamp' in filtered_df.columns:
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & 
                                     (filtered_df['timestamp'] <= end_date)]
    
    # Select numeric columns
    numeric_df = filtered_df.select_dtypes(include=['number'])
    
    # Keep only key metrics
    key_cols = ['altitude', 'temperature', 'bit_flips_count']
    key_cols.extend([col for col in ['cosmic_intensity', 'pressure', 'max_run_length'] 
                     if col in numeric_df.columns])
    
    numeric_df = numeric_df[key_cols]
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation")
    ))
    
    # Add correlation values as text
    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=corr_matrix.columns[j],
                y=corr_matrix.columns[i],
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(color='white' if abs(val) > 0.5 else 'black')
            ))
    
    fig.update_layout(
        title="Parameter Correlation Matrix",
        annotations=annotations,
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# Callback for SEU vs altitude chart
@app.callback(
    Output("seu-altitude-chart", "figure"),
    [Input("apply-date-button", "n_clicks")],
    [State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date")]
)
def update_seu_altitude_chart(n_clicks, start_date, end_date):
    global historical_df
    
    if historical_df is None or historical_df.empty:
        return go.Figure().update_layout(title="No historical data available")
    
    # Filter by date range
    filtered_df = historical_df.copy()
    if 'timestamp' in filtered_df.columns:
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & 
                                     (filtered_df['timestamp'] <= end_date)]
    
    # Check for required columns
    if 'altitude' not in filtered_df.columns or 'bit_flips_count' not in filtered_df.columns:
        return go.Figure().update_layout(title="Required columns not available")
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=filtered_df['altitude'],
        y=filtered_df['bit_flips_count'],
        mode='markers',
        marker=dict(
            size=7,
            opacity=0.6,
            color=filtered_df['temperature'] if 'temperature' in filtered_df.columns else None,
            colorscale='Viridis',
            showscale=True if 'temperature' in filtered_df.columns else False,
            colorbar=dict(title="Temperature (°C)")
        ),
        name='SEU Events'
    ))
    
    # Add trend line
    z = np.polyfit(filtered_df['altitude'], filtered_df['bit_flips_count'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(filtered_df['altitude'].min(), filtered_df['altitude'].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=p(x_range),
        mode='lines',
        name='Trend',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title="SEU Events vs Altitude",
        xaxis_title="Altitude (m)",
        yaxis_title="SEU Count",
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Callback for SEU vs temperature chart
@app.callback(
    Output("seu-temp-chart", "figure"),
    [Input("apply-date-button", "n_clicks")],
    [State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date")]
)
def update_seu_temp_chart(n_clicks, start_date, end_date):
    global historical_df
    
    if historical_df is None or historical_df.empty:
        return go.Figure().update_layout(title="No historical data available")
    
    # Filter by date range
    filtered_df = historical_df.copy()
    if 'timestamp' in filtered_df.columns:
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & 
                                     (filtered_df['timestamp'] <= end_date)]
    
    # Check for required columns
    if 'temperature' not in filtered_df.columns or 'bit_flips_count' not in filtered_df.columns:
        return go.Figure().update_layout(title="Required columns not available")
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=filtered_df['temperature'],
        y=filtered_df['bit_flips_count'],
        mode='markers',
        marker=dict(
            size=7,
            opacity=0.6,
            color=filtered_df['altitude'] if 'altitude' in filtered_df.columns else None,
            colorscale='Cividis',
            showscale=True if 'altitude' in filtered_df.columns else False,
            colorbar=dict(title="Altitude (m)")
        ),
        name='SEU Events'
    ))
    
    # Add trend line
    z = np.polyfit(filtered_df['temperature'], filtered_df['bit_flips_count'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(filtered_df['temperature'].min(), filtered_df['temperature'].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=p(x_range),
        mode='lines',
        name='Trend',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title="SEU Events vs Temperature",
        xaxis_title="Temperature (°C)",
        yaxis_title="SEU Count",
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Callback for geographic map
@app.callback(
    Output("geo-map", "figure"),
    [Input("apply-date-button", "n_clicks")],
    [State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date")]
)
def update_geo_map(n_clicks, start_date, end_date):
    global historical_df
    
    if historical_df is None or historical_df.empty:
        return go.Figure().update_layout(title="No historical data available")
    
    # Filter by date range
    filtered_df = historical_df.copy()
    if 'timestamp' in filtered_df.columns:
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & 
                                     (filtered_df['timestamp'] <= end_date)]
    
    # Check for required columns
    if 'latitude' not in filtered_df.columns or 'longitude' not in filtered_df.columns:
        # Generate fake coordinates based on center point
        base_lat, base_lon = 39.0, -98.0  # Center of USA
        filtered_df['latitude'] = base_lat + np.random.normal(0, 1, len(filtered_df))
        filtered_df['longitude'] = base_lon + np.random.normal(0, 1, len(filtered_df))
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter map
    fig.add_trace(go.Scattergeo(
        lat=filtered_df['latitude'],
        lon=filtered_df['longitude'],
        mode='markers',
        marker=dict(
            size=filtered_df['bit_flips_count'] + 5 if 'bit_flips_count' in filtered_df.columns else 10,
            opacity=0.7,
            color=filtered_df['altitude'] if 'altitude' in filtered_df.columns 
                  else filtered_df['bit_flips_count'] if 'bit_flips_count' in filtered_df.columns 
                  else None,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Altitude (m)" if 'altitude' in filtered_df.columns else "SEU Count")
        ),
        text=[f"Altitude: {a:.0f}m<br>SEUs: {s}<br>Temp: {t:.1f}°C" 
              for a, s, t in zip(filtered_df['altitude'] if 'altitude' in filtered_df.columns else [0]*len(filtered_df),
                               filtered_df['bit_flips_count'] if 'bit_flips_count' in filtered_df.columns else [0]*len(filtered_df),
                               filtered_df['temperature'] if 'temperature' in filtered_df.columns else [0]*len(filtered_df))],
        name='SEU Events'
    ))
    
    # Add flight path
    fig.add_trace(go.Scattergeo(
        lat=filtered_df['latitude'],
        lon=filtered_df['longitude'],
        mode='lines',
        line=dict(
            width=1,
            color='rgba(0,0,255,0.5)'
        ),
        name='Flight Path'
    ))
    
    # Update layout
    fig.update_layout(
        title="SEU Events Geographical Distribution",
        geo=dict(
            scope='usa',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            showsubunits=True,
            subunitcolor='rgb(176, 176, 176)'
        ),
        template="plotly_white",
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting SRAM SEU Detector Dashboard")
    print("="*60)
    print("Dashboard URL: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # For development mode
    app.run_server(debug=True)
