import tensorflow as tf


def custom_loss_function(gps_coordinates, predicted_coordinates):
    """
    Custom loss function to calculate the Mean Squared Error (MSE) based on the provided equations.

    Parameters:
    gps_coordinates: Tensor of actual GPS UTM coordinates, shape (batch_size, 2).
    predicted_coordinates: Tensor of predicted UTM coordinates from the radar, shape (batch_size, 2).

    Returns:
    Mean Squared Error loss.
    """
    utm_x_gps, utm_y_gps = gps_coordinates[:, 0], gps_coordinates[:, 1]
    utm_x_pred, utm_y_pred = predicted_coordinates

    squared_diff_x = tf.square(utm_x_pred - utm_x_gps)
    squared_diff_y = tf.square(utm_y_pred - utm_y_gps)

    mse_loss = tf.reduce_mean(squared_diff_x + squared_diff_y)
    return mse_loss
