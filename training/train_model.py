import tensorflow as tf
from models.radar_calibration_model import RadarCalibrationModel
from training.custom_training_loop import train_step
from utils.data_loader import load_data

from .custom_training_loop import train_step
import tensorflow as tf


def train_model(model, optimizer, radar_data, gps_data, epochs=100):
    """
    Trains the model using the custom training loop.

    Parameters:
        model: The neural network model.
        optimizer: The optimizer to use for training.
        radar_data: Radar input data.
        gps_data: GPS target data.
        epochs (int): Number of training epochs.

    Returns:
        loss_history (list): List of loss values recorded during training.
    """
    loss_history = []

    for epoch in range(epochs):
        loss = train_step(model, optimizer, radar_data, gps_data)
        loss_history.append(loss.numpy())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.numpy()}')

    return loss_history
