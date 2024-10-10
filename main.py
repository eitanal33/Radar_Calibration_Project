from training.train_model import train_model
from models.radar_calibration_model import RadarCalibrationModel
from utils.data_loader import load_data
from utils.visualization import plot_loss_history, plot_predicted_vs_actual_gps
from utils.logger import get_logger, log_experiment_header  # Import the logger functions
from tensorflow import  keras
import tensorflow as tf
import threading
import h5py


def train():
    # Define the number of epochs for the experiment
    epochs = 1000
    import tensorflow as tf
    print(f'  1 {tf.keras.__version__} 2')

    # Initialize the logger
    logger = get_logger()

    # Log the experiment header with the date, time, and number of epochs at the very start
    log_experiment_header(logger, epochs)

    # Load data
    radar_data, gps_data = load_data()

    # Initialize model and optimizer
    model = RadarCalibrationModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    #dropout = tf.keras.layers.Dropout(rate=0.5)  # Add dropout to prevent overfitting
    #print(f"summary: {model.summary()}end of summary")
    # Log initial delta values before training
    delta_layer = model.delta_layer
    logger.info("Initial delta values:")
    logger.info(f"delta_utm_x: {delta_layer.delta_utm_x.numpy()}")
    logger.info(f"delta_utm_y: {delta_layer.delta_utm_y.numpy()}")
    logger.info(f"delta_az: {delta_layer.delta_az.numpy()}")

    # Train the model and get the loss history, predicted GPS, and actual GPS
    loss_history, predicted_gps, actual_gps = train_model(model, optimizer, radar_data, gps_data, epochs=epochs)

    # Log final delta values after training
    logger.info("Final delta values:")
    logger.info(f"delta_utm_x: {delta_layer.delta_utm_x.numpy()}")
    logger.info(f"delta_utm_y: {delta_layer.delta_utm_y.numpy()}")
    logger.info(f"delta_az: {delta_layer.delta_az.numpy()}")
    model.save_weights('model_weights.h5')



    # Visualize the training loss
    plot_loss_history(loss_history)

    # Create a new thread for plotting predicted vs. actual GPS coordinates
    threading.Thread(target=plot_predicted_vs_actual_gps, args=(predicted_gps, actual_gps)).start()


if __name__ == "__main__":
    train()
