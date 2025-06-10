import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

try:
    # Load dataset
    df = pd.read_csv("seu_synthetic_data.csv")
    
    # Show column names for debugging
    print("Available columns:", df.columns.tolist())
    
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
except Exception as e:
    print(f"Error loading data: {e}")
    # Create an empty DataFrame with the required columns if file loading fails
    df = pd.DataFrame(columns=['timestamp', 'altitude', 'temperature', 
                              'sram_region_1_errors', 'sram_region_2_errors',
                              'sram_region_3_errors', 'cosmic_intensity'])

# Dash app setup
app = dash.Dash(__name__)
app.title = "SRAM SEU Event Dashboard"

# Layout
app.layout = html.Div(style={'padding': '20px', 'font-family': 'Arial'}, children=[
    html.H1("🔬 SRAM SEU Event Dashboard", style={'textAlign': 'center'}),    html.Div([
        html.H3("📈 SRAM Errors Over Time"),
        dcc.Graph(
            id='bit-flips-over-time',
            figure=px.line(df, x='timestamp', y=['sram_region_1_errors', 'sram_region_2_errors', 'sram_region_3_errors'], 
                          title='SRAM Errors Over Time')
        )
    ]),

    html.Div([
        html.H3("📡 Altitude Over Time"),
        dcc.Graph(
            id='altitude-over-time',
            figure=px.line(df, x='timestamp', y='altitude', title='Altitude Over Time')
        )
    ]),    html.Div([
        html.H3("🧭 Cosmic Intensity vs Temperature"),
        dcc.Graph(
            id='anomaly-score',
            figure=px.scatter(df, x='temperature', y='cosmic_intensity', color='altitude',
                              size='sram_region_1_errors', title='Cosmic Intensity vs Temperature')
        )
    ]),    html.Div([
        html.H3("📊 Altitude vs SRAM Errors"),
        dcc.Graph(
            id='altitude-vs-errors',
            figure=px.scatter(df, x='altitude', y='sram_region_1_errors',
                              color='cosmic_intensity', size='sram_region_1_errors',
                              title='Altitude vs SRAM Errors')
        )
    ])
])

# Run server
if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Dash server - SEU Detector Dashboard")
    print("="*50)
    print("Dashboard URL: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    try:
        # For newer Dash versions (>=2.0.0), use app.run instead of app.run_server
        app.run(debug=True)
    except Exception as e:
        print(f"Error running Dash server: {e}")
