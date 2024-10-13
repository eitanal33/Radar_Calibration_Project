import tensorflow as tf
from .delta_layer import DeltaLayer

class RadarCalibrationModel(tf.keras.Model):
    def __init__(self,radar_data,track_data):
        super(RadarCalibrationModel, self).__init__()
        self.radar_data = radar_data
        self.track_data = track_data
        self.delta_layer = DeltaLayer(radar_data,track_data)
        #self.dense_layer = tf.keras.layers.Dense(units=64, activation='relu')  # Increase units

    def call(self, inputs,radar_data,track_data):
        utm_x_track, utm_y_track = self.delta_layer(inputs)
        return utm_x_track, utm_y_track
