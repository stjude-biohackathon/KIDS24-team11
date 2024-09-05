import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
from scipy.signal import savgol_filter as sgf

def add_noise(fig, df):
    """Still its own function for ease of use for now, but doesn't need to be."""
    y = np.concatenate((df['transformed'] + df['noise'].values, df['transformed'].values[::-1] - df['noise'].values[::-1]),axis = None)
    x = np.concatenate((df['location'].values,df['location'].values[::-1]),axis = None)
    fig.add_trace(go.Scatter(x=x, y=y, name='Smoothed Noise', fill='toself', line = dict(color='rgba(0,0,0,0)'), fillcolor='rgba(255,0,0,0.2)'))
    return fig

def visualize_data(df, markers = dict(size=5, opacity=0.75), noise_settings = {'window': 100, 'polyorder': 3}):
    if 'location' not in df.columns:
        df['location'] = np.arange(len(df))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['location'], y=df['raw'], name='Raw Signal', mode='markers',marker=markers))
    fig.add_trace(go.Scatter(x=df['location'], y=df['transformed'], name='Transformed Signal'))
    if 'noise' not in df.columns:
        df['noise'] = sgf(abs(df['transformed'] - df['raw']), noise_settings['window'], noise_settings['polyorder'])
        
    fig = add_noise(fig, df)
    
    return fig
        
