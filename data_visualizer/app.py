import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px
import os
import sys
from pathlib import Path

# Get path to project root to help find the data file
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent  # Parent directory of data_visualizer

# Possible data file locations
data_paths = [
    os.path.join(current_dir, "seu_synthetic_data.csv"),  # In current directory
    os.path.join(project_root, "ml_pipeline", "seu_synthetic_data.csv"),  # In ml_pipeline directory
    os.path.join(project_root, "seu_synthetic_data.csv")  # In project root
]

# Try to load the data from one of the possible locations
df = None
for data_path in data_paths:
    try:
        if os.path.exists(data_path):
            print(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)
            
            # Show column names for debugging
            print("Available columns:", df.columns.tolist())
            
            # If we have timestamp column, parse it
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                # If no timestamp but has time_seconds, create timestamps
                if 'time_seconds' in df.columns:
                    # Create a timestamp column using time_seconds
                    start_time = pd.Timestamp.now() - pd.Timedelta(seconds=df['time_seconds'].max())
                    df['timestamp'] = [start_time + pd.Timedelta(seconds=s) for s in df['time_seconds']]
            
            break  # Exit the loop if we successfully loaded the data
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")

# If we couldn't load data from any location
if df is None:
    print("Could not find synthetic data file in any expected location.")
    
    # Create an empty DataFrame with the required columns if file loading fails
    df = pd.DataFrame(columns=['timestamp', 'altitude', 'temperature', 
                              'sram_region_1_errors', 'sram_region_2_errors',
                              'sram_region_3_errors', 'cosmic_intensity'])

# Define functions to create charts based on available columns
def create_errors_time_chart(df):
    """Create SRAM errors over time chart based on available columns"""
    if 'sram_region_1_errors' in df.columns:
        # Use the regional SRAM error columns if they exist
        error_columns = [col for col in df.columns if 'sram_region' in col and 'error' in col]
        return px.line(df, x='timestamp', y=error_columns, title='SRAM Errors Over Time')
    elif 'bit_flips_count' in df.columns:
        # Use the bit_flips_count column from the synthetic data
        return px.line(df, x='timestamp', y='bit_flips_count', title='SRAM Bit Flips Over Time')
    else:
        # Create an empty figure if no suitable columns exist
        return px.line(title="No SRAM error data available")

def create_altitude_chart(df):
    """Create altitude over time chart"""
    if 'altitude' in df.columns:
        return px.line(df, x='timestamp', y='altitude', title='Altitude Over Time')
    else:
        return px.line(title="No altitude data available")

def create_cosmic_temp_chart(df):
    """Create cosmic intensity vs temperature chart"""
    if 'temperature' in df.columns:
        if 'cosmic_intensity' in df.columns:
            size_col = 'sram_region_1_errors' if 'sram_region_1_errors' in df.columns else 'bit_flips_count'
            return px.scatter(df, x='temperature', y='cosmic_intensity', color='altitude',
                          size=size_col if size_col in df.columns else None, 
                          title='Cosmic Intensity vs Temperature')
        else:
            # Try to find alternative columns
            return px.scatter(df, x='temperature', y='altitude', 
                          title='Temperature vs Altitude')
    else:
        return px.scatter(title="No temperature or cosmic intensity data available")

def create_altitude_errors_chart(df):
    """Create altitude vs errors chart"""
    if 'altitude' in df.columns:
        error_col = 'sram_region_1_errors' if 'sram_region_1_errors' in df.columns else 'bit_flips_count'
        if error_col in df.columns:
            color_col = 'cosmic_intensity' if 'cosmic_intensity' in df.columns else 'temperature'
            return px.scatter(df, x='altitude', y=error_col,
                          color=color_col if color_col in df.columns else None, 
                          size=error_col if error_col in df.columns else None,
                          title='Altitude vs SRAM Errors')
        else:
            return px.scatter(df, x='altitude', y='temperature',
                          title='Altitude vs Temperature')
    else:
        return px.scatter(title="No altitude or error data available")

# Dash app setup
app = dash.Dash(__name__)
app.title = "SRAM SEU Event Dashboard"

# Layout
app.layout = html.Div(style={'padding': '20px', 'font-family': 'Arial'}, children=[
    html.H1("SRAM SEU Event Dashboard", style={'textAlign': 'center'}),    html.Div([
        html.H3("SRAM Errors Over Time"),
        dcc.Graph(
            id='bit-flips-over-time',
            figure=create_errors_time_chart(df)
        )
    ], style={'marginBottom': '40px'}),

    html.Div([
        html.H3("Altitude Over Time"),
        dcc.Graph(
            id='altitude-over-time',
            figure=create_altitude_chart(df)
        )
    ], style={'marginBottom': '40px'}),    html.Div([
        html.H3("Cosmic Intensity vs Temperature"),
        dcc.Graph(
            id='cosmic-temperature',
            figure=create_cosmic_temp_chart(df)
        )
    ], style={'marginBottom': '40px'}),    html.Div([
        html.H3("Altitude vs SRAM Errors"),
        dcc.Graph(
            id='altitude-vs-errors',
            figure=create_altitude_errors_chart(df)
        )
    ]),    html.Div([
        html.H3("Data Table"),
        dcc.Store(id='df-store', data=df.to_dict('records')),
        html.Button("Refresh Data", id="refresh-data-btn", n_clicks=0,
                   style={'backgroundColor': '#007bff', 'color': 'white', 'padding': '10px 15px',
                          'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
        html.Div(id="data-table-container")
    ])
])

# Callbacks
@app.callback(
    dash.dependencies.Output('data-table-container', 'children'),
    [dash.dependencies.Input('df-store', 'data'),
     dash.dependencies.Input('refresh-data-btn', 'n_clicks')]
)
def update_data_table(df_data, n_clicks):
    """Update the data table display"""
    if df_data is not None:
        df = pd.DataFrame.from_records(df_data)
        
        # Limit the number of rows displayed for performance
        return [
            html.Div([
                html.H4("Data Preview", style={'marginBottom': '10px'}),
                dcc.Graph(
                    id='data-table-graph',
                    figure=px.scatter(df, x='timestamp', y='sram_region_1_errors', 
                                  title='Data Table Preview - Click and Drag to Zoom')
                ),
                html.Div(f"Showing {len(df)} records", style={'marginTop': '10px'})
            ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px'})
        ]
    else:
        return [html.Div("No data available", style={'color': 'red'})]

@app.callback(
    [dash.dependencies.Output('data-source', 'children'),
     dash.dependencies.Output('data-points', 'children'),
     dash.dependencies.Output('generate-status', 'children')],
    [dash.dependencies.Input('generate-data-btn', 'n_clicks')]
)
def generate_synthetic_data(n_clicks):
    """Generate new synthetic data and update the display"""
    if n_clicks > 0:
        # Generate synthetic data
        import numpy as np
        np.random.seed(0)  # For reproducibility
        
        # Create a DataFrame with synthetic data
        df_synthetic = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='H'),
            'altitude': np.random.uniform(low=1000, high=5000, size=100),
            'temperature': np.random.uniform(low=-50, high=50, size=100),
            'sram_region_1_errors': np.random.poisson(lam=5, size=100),
            'sram_region_2_errors': np.random.poisson(lam=3, size=100),
            'sram_region_3_errors': np.random.poisson(lam=1, size=100),
            'cosmic_intensity': np.random.exponential(scale=1.0, size=100)
        })
        
        # Update the global df variable
        global df
        df = df_synthetic
        
        # Update the df-store with new data
        dash.callback_context.outputs_list[0].set_data(df.to_dict('records'))
        
        return [
            "Currently displaying: Synthetic Data",
            f"Number of data points: {len(df)}",
            "Synthetic data generated successfully!"
        ]
    else:
        return [
            "Currently displaying: Synthetic Data" if df is not None else "No data available",
            f"Number of data points: {len(df) if df is not None else 0}",
            ""
        ]

# Add callback for generating synthetic data
from dash.dependencies import Input, Output, State
import subprocess
import time

@app.callback(
    [Output("data-source", "children"),
     Output("data-points", "children"),
     Output("generate-status", "children"),
     Output("bit-flips-over-time", "figure"),
     Output("altitude-over-time", "figure"),
     Output("cosmic-temperature", "figure"),
     Output("altitude-vs-errors", "figure")],
    [Input("generate-data-btn", "n_clicks")],
    prevent_initial_call=True
)
def generate_new_synthetic_data(n_clicks):
    if n_clicks > 0:
        status_msg = html.Div("Generating new synthetic data...", style={'color': 'blue'})
        
        try:
            # Get the path to the synthetic data generator
            generator_path = os.path.join(project_root, "ml_pipeline", "SEU-Synthetic-Data-Creator.py")
            
            if os.path.exists(generator_path):
                # Run the synthetic data creator script
                subprocess.run([sys.executable, generator_path], check=True, 
                               cwd=os.path.join(project_root, "ml_pipeline"))
                
                # Wait a moment for file to be written
                time.sleep(1)
                
                # Reload the synthetic data
                try:
                    new_df = pd.read_csv(os.path.join(project_root, "ml_pipeline", "seu_synthetic_data.csv"))
                    
                    # If we have timestamp column, parse it
                    if 'timestamp' in new_df.columns:
                        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
                    else:
                        # If no timestamp but has time_seconds, create timestamps
                        if 'time_seconds' in new_df.columns:
                            # Create a timestamp column using time_seconds
                            start_time = pd.Timestamp.now() - pd.Timedelta(seconds=new_df['time_seconds'].max())
                            new_df['timestamp'] = [start_time + pd.Timedelta(seconds=s) for s in new_df['time_seconds']]
                    
                    # Create updated charts
                    errors_chart = create_errors_time_chart(new_df)
                    altitude_chart = create_altitude_chart(new_df)
                    cosmic_chart = create_cosmic_temp_chart(new_df)
                    altitude_errors_chart = create_altitude_errors_chart(new_df)
                    
                    # Copy the file to the data_visualizer directory for future use
                    try:
                        source_path = os.path.join(project_root, "ml_pipeline", "seu_synthetic_data.csv")
                        dest_path = os.path.join(current_dir, "seu_synthetic_data.csv")
                        import shutil
                        shutil.copy2(source_path, dest_path)
                    except Exception as e:
                        print(f"Error copying data file: {e}")
                    
                    return (
                        f"Currently displaying: Newly Generated Synthetic Data",
                        f"Number of data points: {len(new_df)}",
                        html.Div("✅ New synthetic data generated successfully!", style={'color': 'green'}),
                        errors_chart,
                        altitude_chart,
                        cosmic_chart,
                        altitude_errors_chart
                    )
                except Exception as e:
                    print(f"Error loading newly generated data: {e}")
                    return (
                        f"Currently displaying: Existing Synthetic Data",
                        f"Number of data points: {len(df) if df is not None else 0}",
                        html.Div(f"❌ Error loading new data: {str(e)}", style={'color': 'red'}),
                        create_errors_time_chart(df),
                        create_altitude_chart(df),
                        create_cosmic_temp_chart(df),
                        create_altitude_errors_chart(df)
                    )
            else:
                return (
                    f"Currently displaying: Existing Synthetic Data",
                    f"Number of data points: {len(df) if df is not None else 0}",
                    html.Div(f"❌ Error: Couldn't find SEU-Synthetic-Data-Creator.py", style={'color': 'red'}),
                    create_errors_time_chart(df),
                    create_altitude_chart(df),
                    create_cosmic_temp_chart(df),
                    create_altitude_errors_chart(df)
                )
        except Exception as e:
            return (
                f"Currently displaying: Existing Synthetic Data",
                f"Number of data points: {len(df) if df is not None else 0}",
                html.Div(f"❌ Error generating data: {str(e)}", style={'color': 'red'}),
                create_errors_time_chart(df),
                create_altitude_chart(df),
                create_cosmic_temp_chart(df),
                create_altitude_errors_chart(df)
            )
    
    # Default return if button not clicked
    return (
        f"Currently displaying: Synthetic Data",
        f"Number of data points: {len(df) if df is not None else 0}",
        html.Div("Click the button to generate new synthetic data", style={'color': 'gray'}),
        create_errors_time_chart(df),
        create_altitude_chart(df),
        create_cosmic_temp_chart(df),
        create_altitude_errors_chart(df)
    )

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
