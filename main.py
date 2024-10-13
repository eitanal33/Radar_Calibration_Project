from training.train_model import train_model
from models.radar_calibration_model import RadarCalibrationModel
from utils.data_loader import load_data
from utils.visualization import plot_loss_history, plot_predicted_vs_actual_gps
from utils.logger import get_logger, log_experiment_header
import tensorflow as tf
import threading


def train():
    # Define the number of epochs for the experiment
    epochs = 1000
    print(f'TensorFlow version: {tf.__version__}')

    # Initialize the logger
    logger = get_logger()

    # Log the experiment header with the date, time, and number of epochs at the very start
    log_experiment_header(logger, epochs)

    # Load data
    radar_data, gps_data, track_data, false_radar_data = load_data()  # Updated to use the new data loader

    # Initialize model and optimizer
    model = RadarCalibrationModel(track_data)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Log initial delta values before training
    delta_layer = model.delta_layer
    logger.info("Initial delta values:")
    logger.info(f"delta_utm_x: {delta_layer.delta_utm_x.numpy()}")
    logger.info(f"delta_utm_y: {delta_layer.delta_utm_y.numpy()}")
    logger.info(f"delta_az: {delta_layer.delta_az.numpy()}")

    # Train the model and get the loss history, predicted GPS, and actual GPS
    loss_history, predicted_gps, actual_gps = train_model(model, optimizer, false_radar_data,radar_data,track_data, gps_data, epochs=epochs)

    # Log final delta values after training
    logger.info("Final delta values:")
    logger.info(f"delta_utm_x: {delta_layer.delta_utm_x.numpy()}")
    logger.info(f"delta_utm_y: {delta_layer.delta_utm_y.numpy()}")
    logger.info(f"delta_az: {delta_layer.delta_az.numpy()}")

    # Save model weights
    model.save_weights('model_weights.h5')

    # Visualize the training loss
    plot_loss_history(loss_history)

    # Create a new thread for plotting predicted vs. actual GPS coordinates
    threading.Thread(target=plot_predicted_vs_actual_gps, args=(predicted_gps, actual_gps)).start()


if __name__ == "__main__":
    train()
