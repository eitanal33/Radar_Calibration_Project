import tensorflow as tf


def custom_loss_function(utm_x_track, utm_y_track, utm_x_gps, utm_y_gps):
    n = tf.cast(tf.size(utm_x_track), tf.float32)  # Get the number of samples
    loss = tf.reduce_mean(tf.sqrt(
        tf.square(utm_x_track - utm_x_gps) + tf.square(utm_y_track - utm_y_gps)
    ))

    return loss


def train_step(model, optimizer, false_radar_data, gps_labels,radar_data,track_data):
    # Unpack the false radar data into UTM coordinates
    utm_x_radar_false, utm_y_radar_false = false_radar_data[:, 0], false_radar_data[:, 1]

    with tf.GradientTape() as tape:
        # Forward pass to get the predicted UTM coordinates for the track
        utm_x_track, utm_y_track = model(false_radar_data,radar_data,track_data)

        # Compute the loss using the false radar data and GPS labels
        loss = custom_loss_function(utm_x_track, utm_y_track, gps_labels[0], gps_labels[1], utm_x_radar_false, utm_y_radar_false)

    # Compute the gradients of the loss with respect to the model's trainable variables
    gradients = tape.gradient(loss, model.trainable_variables)

    if not gradients or all(g is None for g in gradients):
        print("Warning: Gradients are not being computed correctly. Check model connections.")

    # Apply the computed gradients to the model's variables using the optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

