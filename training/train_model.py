import tensorflow as tf
from models.radar_calibration_model import RadarCalibrationModel
from training.custom_training_loop import train_step
from utils.data_loader import load_data
import os
import threading
import datetime
import time
import numpy as np
import cv2

from utils.visualization import plot_tracks_as_image
from .custom_training_loop import train_step


def train_model(model, optimizer, radar_data, gps_data, epochs=100, callbacks=None):
    """Trains the model using the custom training loop."""
    loss_history = []

    model_name = time.time()
    train_log_dir = f'logs/fit/{model_name}'  # Fixed typo from 'fits' to 'fit'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    gps_utm_x, gps_utm_y = gps_data

    # Create images directory for storing plot images
    os.makedirs('images', exist_ok=True)

    # Create a TensorBoard writer for logging
    if callbacks:
        for callback in callbacks:
            if hasattr(callback, 'set_model'):
                callback.set_model(model)  # Ensure the callback knows the model

    for epoch in range(epochs):
        loss, deltas, utm_x_plot, utm_y_plot = train_step(model, optimizer, radar_data, gps_data)
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

            # Log track visualization as an image
            gps_track = np.column_stack((gps_utm_x, gps_utm_y))
            radar_track = np.column_stack((utm_x_plot, utm_y_plot))
            track_image = plot_tracks_as_image(gps_track, radar_track)
            tf.summary.image('GPS vs Radar Track', track_image, step=epoch)

            if track_image.ndim == 4:
                track_image = np.squeeze(track_image, axis=0)

            # Save the plot as an image
            track_image_path = f'images/epoch_{epoch}.png'
            tf.keras.preprocessing.image.save_img(track_image_path, track_image)

            print(f'Epoch {epoch}: Loss = {loss.numpy()}')

    # Compile images into a video
    compile_video()

    return loss_history


def compile_video():
    """Compiles saved images into a video."""
    frame_size = (640, 480)  # Size of the video frame
    fps = 10  # Frames per second
    video_filename = "tensorboard_video_{}.mp4".format(
        datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H-%M-%S"))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    # Write images to video
    for epoch in range(2300):  # Adjust range according to your epochs
        img_path = f'images/epoch_{epoch}.png'
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, frame_size)  # Resize to fit video frame
            out.write(img)  # Write the frame

    out.release()
    cv2.destroyAllWindows()
    print(f'Video saved as: {video_filename}')
