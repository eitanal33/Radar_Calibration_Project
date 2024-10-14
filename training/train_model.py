import tensorflow as tf
from models.radar_calibration_model import RadarCalibrationModel
from training.custom_training_loop import train_step
from utils.data_loader import load_data
import os
import threading

from .custom_training_loop import train_step
import tensorflow as tf


def train_model(model, optimizer, radar_data, gps_data, epochs=100, callbacks=None):
    """
    Trains the model using the custom training loop.

    Parameters:
        model: The neural network model.
        optimizer: The optimizer to use for training.
        radar_data: Radar input data.
        gps_data: GPS target data.
        epochs (int): Number of training epochs.
        callbacks (list): List of callbacks to use during training (e.g., TensorBoard).

    Returns:
        loss_history (list): List of loss values recorded during training.
    """
    # Ensure callbacks are handled properly
    if callbacks is None:
        callbacks = []

    loss_history = []

    for epoch in range(epochs):
        loss = train_step(model, optimizer, radar_data, gps_data)
        loss_history.append(loss.numpy())

        # Log the loss to TensorBoard
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs={'loss': loss.numpy()})

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.numpy()}')

    return loss_history
