import matplotlib.pyplot as plt

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
