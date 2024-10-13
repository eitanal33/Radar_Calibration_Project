import numpy as np
import pandas as pd


def calculate_alpha_and_range(utm_x_radar, utm_y_radar,az_radar, plot_data):
    """
    Calculate the alpha (angle) and range between the radar and plot coordinates.

    Parameters:
    - utm_x_radar (float): UTM x-coordinate of the radar
    - utm_y_radar (float): UTM y-coordinate of the radar
    - plot_data (DataFrame): DataFrame containing 'utm_x_plot' and 'utm_y_plot'

    Returns:
    - plot_data (DataFrame): Original plot_data with additional columns for calculated range and alpha
    """
    # Calculate the distance (range) and angle (alpha)
    plot_data['calculated_range'] = np.sqrt((plot_data['utm_x_plot'] - utm_x_radar) ** 2 +
                                            (plot_data['utm_y_plot'] - utm_y_radar) ** 2)

    plot_data['calculated_alpha'] = np.arctan2(plot_data['utm_y_plot'] - utm_y_radar,
                                               plot_data['utm_x_plot'] - utm_x_radar)

    # Convert radians to degrees for alpha
    plot_data['calculated_alpha'] = np.degrees(plot_data['calculated_alpha'])
    plot_data['az_track'] = plot_data['calculated_alpha'] + az_radar
    return plot_data


def main():
    # Example wrong radar data (constant)
    utm_x_radar = 1020.0
    utm_y_radar = 1985.0
    az_radar = -30.0

    # Load the plot data from CSV or create it for testing
    plot_data = pd.DataFrame({
        'utm_x_plot': [1020.7411170124064, 1021.6371931745533, 1022.7140952243667],
        'utm_y_plot': [1985.5852555485126, 1986.0116936851182, 1986.1289835920002]
    })

    # Calculate alpha and range
    plot_data = calculate_alpha_and_range(utm_x_radar, utm_y_radar,az_radar, plot_data)

    # Print the results
    print(plot_data)


if __name__ == "__main__":
    main()