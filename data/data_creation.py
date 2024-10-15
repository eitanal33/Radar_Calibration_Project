import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

utm_x_radar = 1000
utm_y_radar = 2000
azimuth_radar = 30  # Constant azimuth for the radar in degrees

# Number of GPS points to generate
num_points = 2000

path_length = 100  # Total distance of the path
step_size_mean = 1.0  # Mean step size
step_size_std = 0.2  # Standard deviation for step size variation

# Generate a natural path with realistic direction changes
gps_path_x = [utm_x_radar]
gps_path_y = [utm_y_radar]

# Initial direction angle
current_angle = 0

for _ in range(num_points - 1):
    # Vary the angle slightly to simulate a natural path
    angle_variation = np.random.normal(0, 10)  # Mean 0, Std dev 10 degrees
    current_angle += angle_variation  # Update current angle with variation

    # Get a random step size
    step_size = np.random.normal(step_size_mean, step_size_std)

    # Calculate new GPS coordinates
    new_x = gps_path_x[-1] + step_size * np.cos(np.radians(current_angle))
    new_y = gps_path_y[-1] + step_size * np.sin(np.radians(current_angle))
    gps_path_x.append(new_x)
    gps_path_y.append(new_y)

# Convert to numpy arrays, skipping the initial radar position
utm_x_gps = np.array(gps_path_x[1:])  # True GPS coordinates
utm_y_gps = np.array(gps_path_y[1:])  # True GPS coordinates

# Define known delta values for false radar coordinates
delta_utm_x = 20
delta_utm_y = -15
delta_az = -0.8

# Calculate false radar coordinates based on the radar's true position
utm_x_radar_false = utm_x_radar + delta_utm_x
utm_y_radar_false = utm_y_radar + delta_utm_y

# Track data calculations
range_ = np.sqrt((utm_x_gps - utm_x_radar) ** 2 + (utm_y_gps - utm_y_radar) ** 2)  # Step 1: Calculate range
alpha = np.arctan2(utm_x_gps - utm_x_radar, utm_y_gps - utm_y_radar)  # Step 2: Calculate alpha (in radians)

# Convert alpha to degrees
alpha_degrees = np.degrees(alpha)

# Step 3: Calculate az_track using the formula: az_track = alpha + az_radar
az_track = alpha_degrees + azimuth_radar + delta_az

# Ensure azimuth is within 0-360 degrees
az_track = np.mod(az_track, 360)

# Step 4: Calculate the false UTM coordinates of the plot
utm_x_plot = utm_x_radar_false + range_ * np.sin(np.radians(az_track))
utm_y_plot = utm_y_radar_false + range_ * np.cos(np.radians(az_track))

# Create DataFrames for radar, GPS, and track data
radar_data = pd.DataFrame({
    'utm_x_radar_false': [utm_x_radar_false] * (num_points - 1),
    'utm_y_radar_false': [utm_y_radar_false] * (num_points - 1),
    'r': range_,
    'alpha': alpha_degrees,  # Angle in degrees
    'azimuth_radar': [azimuth_radar] * (num_points - 1),
})

gps_data = pd.DataFrame({
    'utm_x_gps': utm_x_gps,
    'utm_y_gps': utm_y_gps
})

track_data = pd.DataFrame({
    'range': range_,
    'alpha': alpha_degrees,  # Angle in degrees
    'az_track': az_track,
    'utm_x_plot': utm_x_plot,
    'utm_y_plot': utm_y_plot
})



#radar_data.to_csv('radar_data.csv', index=False)
#gps_data.to_csv('gps_data.csv', index=False)
#track_data.to_csv('track_data.csv',index = False)

# Print all the DataFrames
print("Radar Data:")
print(radar_data)

print("\nGPS Data:")
print(gps_data)

print("\nTrack Data:")
print(track_data)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(utm_x_gps, utm_y_gps, marker='o', label='True GPS Path', color='blue', markersize=1)
plt.plot(utm_x_plot, utm_y_plot, marker='x', label='Calculated Plot Path', color='red', markersize=1)
plt.scatter(utm_x_radar, utm_y_radar, color='green', marker='s', label='Radar Position')
plt.scatter(utm_x_radar_false, utm_y_radar_false, color='purple', marker='D', label='False Radar Position')
plt.title('Radar Tracking Visualization')
plt.xlabel('UTM X Coordinate')
plt.ylabel('UTM Y Coordinate')
plt.legend()
plt.grid()
plt.axis('equal')  # Equal scaling for both axes
plt.show()

