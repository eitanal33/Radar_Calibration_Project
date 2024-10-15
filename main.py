from training.train_model import train_model
from models.radar_calibration_model import RadarCalibrationModel
from utils.data_loader import load_data
from utils.visualization import plot_loss_history, plot_predicted_vs_actual_gps
from utils.logger import get_logger, log_experiment_header  # Import the logger functions
from tensorflow import  keras
import tensorflow as tf
import threading
import h5py
import os
from tensorflow.keras.callbacks import TensorBoard


def train():
    epochs = 2500
    log_dir = os.path.join("logs", "fit")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    radar_data, gps_data = load_data()

    model = RadarCalibrationModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # Compile the model with the optimizer
    #model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model and get the loss history
    loss_history = train_model(model, optimizer, radar_data, gps_data, epochs=epochs, callbacks=[tensorboard_callback])

    plot_loss_history(loss_history)


if __name__ == "__main__":
    train()