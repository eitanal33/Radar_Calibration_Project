import tensorflow as tf
from .delta_layer import DeltaLayer

class RadarCalibrationModel(tf.keras.Model):
    def __init__(self):
        super(RadarCalibrationModel, self).__init__()
        self.delta_layer = DeltaLayer()
        #self.dense_layer = tf.keras.layers.Dense(units=64, activation='relu')  # Increase units

    def call(self, inputs):
        utm_x_track, utm_y_track = self.delta_layer(inputs)
        return utm_x_track, utm_y_track
