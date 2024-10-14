import numpy as np
import tensorflow as tf

class DeltaLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DeltaLayer, self).__init__(**kwargs)

        self.delta_utm_x = self.add_weight(name="delta_utm_x", initializer="zeros", trainable=True)
        self.delta_utm_y = self.add_weight(name="delta_utm_y", initializer="zeros", trainable=True)
        self.delta_az = self.add_weight(name="delta_az", initializer="zeros", trainable=True)

    def call(self, inputs):
        utm_x_radar = inputs[:, 0]
        utm_y_radar = inputs[:, 1]
        r = inputs[:, 2]
        alpha = inputs[:, 3]
        az = inputs[:, 4]
        az = np.radians(az)
        alpha = np.radians(alpha)
        utm_x_track = utm_x_radar + self.delta_utm_x + r * tf.sin(alpha + az + self.delta_az)
        utm_y_track = utm_y_radar + self.delta_utm_y + r * tf.cos(alpha + az + self.delta_az)
        #print(f"utm_x_track {utm_x_track[1134]}")
        #print(f"utm_y_track {utm_y_track[1134]}")
        return utm_x_track, utm_y_track
