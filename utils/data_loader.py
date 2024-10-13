import pandas as pd
import numpy as np

def load_data():
    # Radar data
    radar_df = pd.read_csv('data/radar_data.csv')
    radar_data = radar_df[['utm_x_radar', 'utm_y_radar', 'az']].values.astype(np.float32)
    #print(radar_data[0,2])
    # GPS data
    gps_df = pd.read_csv('data/gps_data.csv')
    gps_data = gps_df[['utm_x_gps', 'utm_y_gps']].values.astype(np.float32)

    # False radar data
    false_radar_df = pd.read_csv('data/radar_data_false.csv')  # Use the correct file for false radar data
    false_radar_data = false_radar_df[['utm_x_radar_false', 'utm_y_radar_false']].values.astype(np.float32)

    # Track data
    track_df = pd.read_csv('data/track_data.csv')
    track_data = track_df[['range', 'alpha', 'az_track', 'utm_x_plot', 'utm_y_plot']].values.astype(np.float32)

    return radar_data, gps_data, track_data, false_radar_data


