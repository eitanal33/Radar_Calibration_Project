import tensorflow as tf
from models.radar_calibration_model import RadarCalibrationModel
from training.custom_training_loop import train_step
from utils.data_loader import load_data
import os
import threading
import time

from .custom_training_loop import train_step


def train_model(model, optimizer, radar_data, gps_data, epochs=100, callbacks=None):
    """Trains the model using the custom training loop."""
    loss_history = []

    model_name = time.time()
    train_log_dir = f'logs/fit/{model_name}'  # Fixed typo from 'fits' to 'fit'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Create a TensorBoard writer for logging
    if callbacks:
        for callback in callbacks:
            if hasattr(callback, 'set_model'):
                callback.set_model(model)  # Ensure the callback knows the model

    for epoch in range(epochs):
        loss, deltas = train_step(model, optimizer, radar_data, gps_data)
        loss_history.append(loss.numpy())

        # Log the loss to TensorBoard automatically using callbacks
        if callbacks:
            logs = {'loss': loss.numpy()}
            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

        # Log additional metrics
        with train_summary_writer.as_default():
            # Log the loss value
            tf.summary.scalar('Loss', loss, step=epoch)

            # Log delta correction weights with headings
            utm_x_weights, utm_y_weights, azimuth_weights = deltas
            tf.summary.scalar('UTM_X_Correction', utm_x_weights.numpy(), step=epoch)
            tf.summary.scalar('UTM_Y_Correction', utm_y_weights.numpy(), step=epoch)
            tf.summary.scalar('Azimuth_Correction', azimuth_weights.numpy(), step=epoch)

        # Print the loss at every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.numpy()}')

    return loss_history
