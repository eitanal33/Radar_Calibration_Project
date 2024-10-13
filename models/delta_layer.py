import tensorflow as tf
from distributed.profile import plot_data
import pandas as pd

from utils.alpha_range_calc import calculate_alpha_and_range
class DeltaLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DeltaLayer, self).__init__(**kwargs)
        self.delta_utm_x = self.add_weight(name="delta_utm_x", initializer="zeros", trainable=True)
        self.delta_utm_y = self.add_weight(name="delta_utm_y", initializer="zeros", trainable=True)
        self.delta_az = self.add_weight(name="delta_az", initializer="zeros", trainable=True)

    def call(self, inputs,radar_data, plot_data):
        utm_x_radar = inputs[:, 0]
        utm_y_radar = inputs[:, 1]
        az_radar = radar_data[0,2]
        plot_data = calculate_alpha_and_range(utm_x_radar, utm_y_radar, az_radar, plot_data)        #r = inputs[:, 2]this is where it crushes
        #alpha = inputs[:, 3]
        #az = inputs[:, 4]

        utm_x_track = utm_x_radar + self.delta_utm_x + r * tf.sin(alpha + az + self.delta_az)
        utm_y_track = utm_y_radar + self.delta_utm_y + r * tf.cos(alpha + az + self.delta_az)
        return utm_x_track, utm_y_track
