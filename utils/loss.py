
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


def gradient_penalty(images, generated_images,discriminator):
    
    epsilon = tf.random.uniform([images.shape[0], 1, 1, 1],0.0,1.0)
    x_interpolate= epsilon*images + (1-epsilon) * (generated_images)

    #comute gradient of critic
    with tf.GradientTape() as t:
        t.watch(x_interpolate)
        disc_interpolate=discriminator(x_interpolate)
    gradient = t.gradient(disc_interpolate,x_interpolate)
    norme=tf.sqrt(tf.reduce_sum( gradient ** 2 , axis=[1,2] ) )
    gp=tf.reduce_mean( ( norme - 1.0 ) ** 2 )
    return gp


def wasserstein_discriminator_loss(real_output, fake_output,images,generated_images,discriminator,coeff=10.0):

    gp=gradient_penalty(images, generated_images,discriminator)

    loss= (tf.reduce_mean(real_output)-tf.reduce_mean(fake_output) + coeff * gp)
    return loss

def wasserstein_generator_loss(fake_output):
    loss=  tf.reduce_mean(fake_output)
    return loss