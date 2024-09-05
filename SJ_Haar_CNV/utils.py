import pandas as pd, numpy as np, plotly.express as px
import scipy.stats as sts


"""
I'm just shoving some functions here for now. I'll clean this up later.
"""
def parameterize_wavelet(df):
    df['end'] = df['length'].cumsum()
    df['start'] = df['end'].shift(1).fillna(0)
    return df

def expand_wavelet(df):
    return pd.DataFrame(data = np.repeat(df['value'],df['length']), columns = ['value']).reset_index(drop = True)

def visualize_wavelet(df):
    data = expand_wavelet(df)
    px.line(data, y = 'value',line_shape='hv').show()

