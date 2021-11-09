
import os
import sys

sys.path.append((os.getcwd()))
sys.path.append(os.path.dirname(os.getcwd()))


import datetime
import time
import numpy as np
import tensorflow as tf



import param 

from utils import import_DS 
from utils import loss  
from utils import network_model




dataset , nb_classes=import_DS.importMSTAR() ,9
img_shape=(128,128,3)


generator = network_model.make_generator_conditional_model(nb_classes=nb_classes)
discriminator = network_model.make_discriminator_conditional_model(nb_classes=nb_classes,img_shape=img_shape)

print(generator.summary())
print(discriminator.summary())

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

generator_loss=loss.generator_loss
discriminator_loss=loss.discriminator_loss


#------------------------------------------------------------------------#
print(" def metrics")

discriminator_loss_M = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
discriminator_accuracy_M = tf.keras.metrics.BinaryCrossentropy(name='d_accuracy')
generator_loss_M = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)
generator_accuracy_M = tf.keras.metrics.BinaryCrossentropy('g_accuracy')


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
gen_log_dir = './logs/gradient_tape/' + current_time + '/gen'
disc_log_dir = './logs/gradient_tape/' + current_time + '/disc'
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)

img_log_dir = './logs/gradient_tape/' + current_time + '/img'
img_summary_writer = tf.summary.create_file_writer(img_log_dir)

print(" def metrics\n\n")
#------------------------------------------------------------------------#

#@tf.function
def train_step(images):
    noise = tf.random.normal([param.BATCH_SIZE, param.noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      #images[1].numpy() = label of the images
      generated_images = generator((noise,images[1]), training=True)
      real_output = discriminator((images[0],images[1]), training=True)
      fake_output = discriminator((generated_images,images[1]), training=True)
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    discriminator_loss_M(disc_loss)
    discriminator_accuracy_M(tf.ones_like(real_output),real_output)
    discriminator_accuracy_M(tf.zeros_like(fake_output),fake_output)
    generator_loss_M(gen_loss)

    return (gen_loss,disc_loss)


# to visualize progress in the animated GIF)
seed = (tf.random.normal([9, param.noise_dim]),np.array([0,1,2,3,4,5,6,7,8]))      

def train(dataset, epochs):
    gen_loss,disc_loss =0,0
    for epoch in range(epochs):
        start = time.time()
        
        for image_batch in dataset:
            if image_batch[1].numpy().size == param.BATCH_SIZE :
                (gen_loss,disc_loss) = train_step(image_batch)

        #summary img
        predictions = generator(seed, training=False)
        with img_summary_writer.as_default():
            tf.summary.image('img_generate', predictions, step=epoch)


        with gen_summary_writer.as_default():
            tf.summary.scalar('loss', generator_loss_M.result(), step=epoch)


        with disc_summary_writer.as_default():
            tf.summary.scalar('loss', discriminator_loss_M.result(), step=epoch)
            tf.summary.scalar('accuracy', discriminator_accuracy_M.result(), step=epoch)

        print ('Epoch {} LossG = {} == {}  LossD={} == {} Time for epoch {} sec'
              .format(epoch + 1,gen_loss,generator_loss_M.result(),disc_loss,discriminator_loss_M.result(), time.time()-start))

        # Reset metrics every epoch
        discriminator_loss_M.reset_states()
        generator_loss_M.reset_states()
        discriminator_accuracy_M.reset_states()


    
print( "*** START ***")
train(dataset, param.EPOCHS)