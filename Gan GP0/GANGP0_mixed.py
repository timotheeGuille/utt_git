print("---start---")
print("WDCGAN")

#typeExec = 0(serveur with Gpu) 1(local cpu and low bdd)
typeExec =0

#------------------------------------------------------------------------#
print(" import")


import tensorflow as tf

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import datetime

from IPython import display



print(" import\n\n")
#------------------------------------------------------------------------#
print(" param")



BATCH_SIZE = 256 if typeExec == 0 else 16
EPOCHS = 50 if typeExec == 0 else 5
noise_dim = 100
num_examples_to_generate = 16

print(" param\n\n")
#------------------------------------------------------------------------#
print(" dataset")



#import
(x_train,y_train),(_,_)=tf.keras.datasets.mnist.load_data()

if typeExec == 1:
  x_train=x_train[0:256,:,:]
  y_train=y_train[0:256]


#cut size to avoid size bigger than batchsize
size_cut=x_train.shape[0]-x_train.shape[0]%BATCH_SIZE

x_train=x_train[0:size_cut,:,:]
y_train=y_train[0:size_cut]


#reshape and norm
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train= ( x_train - 127.5 ) / 127.5

#suffle
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(x_train.shape[0]).batch(BATCH_SIZE)



print(" dataset\n\n")
#------------------------------------------------------------------------#
print(" def model")

from IPython import display
#G model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    #assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


#D model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = make_generator_model()
print(generator.summary())
discriminator = make_discriminator_model()
print(discriminator.summary())


print(" def model\n\n")
#------------------------------------------------------------------------#
print(" def loss")


def gradient_penalty00(images, generated_images):

    epsilon = tf.random.uniform([images.shape[0], 1, 1, 1],0.0,1.0)
    x_interpolate= epsilon*images + (1-epsilon) * (generated_images)

    #comute gradient of critic
    with tf.GradientTape() as t:
        t.watch(x_interpolate)
        disc_interpolate=discriminator(x_interpolate)
    gradient = t.gradient(disc_interpolate,x_interpolate)
    norme=tf.sqrt(tf.reduce_sum( gradient ** 2 , axis=[1,2] ) )
    gp=tf.reduce_mean( ( norme ) ** 2 )
    return gp

def gradient_penalty0(images, generated_images):

   

    #comute gradient of critic
    with tf.GradientTape() as t_real,tf.GradientTape() as t_generate:
        t_real.watch(images)
        disc_real=discriminator(images)
        t_generate.watch(images)
        disc_generate=discriminator(generated_images)
    gradient_real = t_real.gradient(disc_real,images)
    norme_real=tf.sqrt(tf.reduce_sum( gradient_real ** 2 , axis=[1,2] ) )

    gradient_generate = t_generate.gradient(disc_generate,generated_images)
    norme_generate=tf.sqrt(tf.reduce_sum( gradient_generate ** 2 , axis=[1,2] ) )

    gp=tf.reduce_mean(  norme_real  )+ tf.reduce_mean(  norme_generate  )
    return gp

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output,gp):
    coeff = 10.0

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss - coeff * gp
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

print(" def loss\n\n")
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
print(" def train")




@tf.function
def  train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gp=gradient_penalty0(images, generated_images)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output,gp)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    discriminator_loss_M(disc_loss)
    discriminator_accuracy_M(real_output,tf.ones_like(real_output))
    generator_loss_M(gen_loss)

    return (gen_loss,disc_loss)

# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])    
    
def train(dataset, epochs):
    gen_loss,disc_loss =0,0
    for epoch in range(epochs):
        start = time.time()
        
        for image_batch in dataset:
            (gen_loss,disc_loss) = train_step(image_batch)

        # Produce images for the GIF
        #display.clear_output(wait=True)
        generate_and_save_images(generator,epoch + 1,seed)

   
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

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           seed)
  
  
def generate_and_save_images(model, epoch, test_input):

  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

  with img_summary_writer.as_default():
      tf.summary.image('img_generate', predictions, step=epoch-1)
  #plt.show()




print(" def train\n\n")
#------------------------------------------------------------------------#
print("train\n")


train(train_dataset, EPOCHS)


print("train\n\n")
#------------------------------------------------------------------------#
print("display")



# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)


anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)


print("display\n\n")
#------------------------------------------------------------------------#
print("---END---")