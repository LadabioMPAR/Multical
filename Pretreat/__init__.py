import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def normalizar(df):
    return (df - df.mean()) / df.std()

def media_movel(df, window_size):
    return df.apply(lambda x: x.rolling(window=window_size, min_periods=1).mean())

def derivada_savitsky_golay(df, window_length, polyorder):
    return df.apply(lambda x: savgol_filter(x, window_length, polyorder, deriv=1))