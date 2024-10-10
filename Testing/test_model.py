import tensorflow as tf
import pandas as pd
import numpy as np
from models.radar_calibration_model import RadarCalibrationModel
from Testing.config import RADAR_DATA_PATH, GPS_DATA_PATH


def load_test_data():
    radar_data = pd.read_csv(RADAR_DATA_PATH)
    gps_data = pd.read_csv(GPS_DATA_PATH)
    radar_inputs = radar_data[['utm_x_radar', 'utm_y_radar', 'r', 'alpha', 'az']].values
    gps_outputs = gps_data[['utm_x_gps', 'utm_y_gps']].values
    return radar_inputs, gps_outputs


def evaluate_model_with_weights():
    # Create and initialize the model
    model = RadarCalibrationModel()
    dummy_input = tf.zeros((1, 5))  # Adjust based on your input dimensions
    model(dummy_input)  # Initialize the model layers

    # Define your hardcoded weights for the DeltaLayer
    delta_weights = np.array([-38.23911285, -16.16359329, 0.055447813])  # All deltas in a single array

    # Set the weights of the DeltaLayer
    model.delta_layer.set_weights([delta_weights[0], delta_weights[1], delta_weights[2]])  # Provide all three weights

    # Load test data
    radar_inputs, gps_outputs = load_test_data()

    # Run the model with the hardcoded weights
    predicted_utm_x, predicted_utm_y = model(radar_inputs)
    predicted_utm_x = predicted_utm_x.numpy()
    predicted_utm_y = predicted_utm_y.numpy()

    print(f"Predicted UTM X: {predicted_utm_x}, Predicted UTM Y: {predicted_utm_y}")
    print(f"Actual GPS Outputs UTM X: {gps_outputs[:, 0]}, UTM Y: {gps_outputs[:, 1]}")

    # Calculate Mean Squared Error (MSE)
    mse_x = np.mean((predicted_utm_x - gps_outputs[:, 0]) ** 2)
    mse_y = np.mean((predicted_utm_y - gps_outputs[:, 1]) ** 2)
    overall_mse = (mse_x + mse_y) / 2

    print(f"Mean Squared Error (UTM X): {mse_x}")
    print(f"Mean Squared Error (UTM Y): {mse_y}")
    print(f"Overall Mean Squared Error: {overall_mse}")


if __name__ == "__main__":
    evaluate_model_with_weights()
