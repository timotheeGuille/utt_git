
import tensorflow as tf


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def accuracy (real_output,fake_output):
    true_P=(real_output.numpy()>0).sum()
    true_N=(fake_output.numpy()<0).sum()
    false_P=(real_output.numpy()<=0).sum()
    false_N=(fake_output.numpy()>=0).sum()

    acc= (true_P + true_N) / (true_P + true_N + false_P + false_N)
    return acc