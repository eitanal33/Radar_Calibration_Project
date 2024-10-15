import matplotlib.pyplot as plt
import io
import tensorflow as tf
def plot_loss_history(loss_history):
    """Plot the loss history over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predicted_vs_actual_gps(predicted_gps, actual_gps):
    """Plot the predicted GPS coordinates vs. actual GPS coordinates."""
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_gps[:, 0], actual_gps[:, 1], label='Actual GPS', color='blue', marker='o')
    plt.scatter(predicted_gps[:, 0], predicted_gps[:, 1], label='Predicted GPS', color='red', marker='x')
    plt.xlabel('UTM X Coordinate')
    plt.ylabel('UTM Y Coordinate')
    plt.title('Predicted vs. Actual GPS Coordinates')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_tracks_as_image(gps_track, radar_track):
    fig, ax = plt.subplots()
    ax.plot(gps_track[:, 0], gps_track[:, 1], color='blue', label='GPS Track')
    ax.plot(radar_track[:, 0], radar_track[:, 1], color='red', label='Radar Track')
    ax.set_xlabel('UTM X Coordinate')
    ax.set_ylabel('UTM Y Coordinate')
    ax.set_title('Radar and GPS Tracking Visualization')
    ax.legend(loc='upper right')

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)  # Add batch dimension
    return image