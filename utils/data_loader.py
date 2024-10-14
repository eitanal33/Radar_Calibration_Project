import pandas as pd
import numpy as np

def load_data():
    radar_df = pd.read_csv('data/radar_data.csv')
    gps_df = pd.read_csv('data/gps_data.csv')

    radar_data = radar_df[['utm_x_radar', 'utm_y_radar', 'r', 'alpha', 'az']].values.astype(np.float32)
    gps_data = gps_df[['utm_x_gps', 'utm_y_gps']].values.astype(np.float32).T

    return radar_data, gps_data
