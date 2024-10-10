import tensorflow as tf


def custom_loss_function(utm_x_track, utm_y_track, utm_x_gps, utm_y_gps):
    n = tf.cast(tf.size(utm_x_track), tf.float32)  # Get the number of samples
    loss = tf.reduce_mean(
        tf.square(utm_x_track - utm_x_gps) + tf.square(utm_y_track - utm_y_gps)
    )
    loss = tf.sqrt(loss)
    return loss


def train_step(model, optimizer, inputs, gps_labels):
    with tf.GradientTape() as tape:
        utm_x_track, utm_y_track = model(inputs)
        loss = custom_loss_function(utm_x_track, utm_y_track, gps_labels[0], gps_labels[1])
    gradients = tape.gradient(loss, model.trainable_variables)

    if not gradients or all(g is None for g in gradients):
        print("Warning: Gradients are not being computed correctly. Check model connections.")

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

