import tensorflow as tf

import numpy as np
import IPython.display as display
import os

import imageio


import datetime
import time


import os
import sys

sys.path.append((os.getcwd()))
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from utils import loss 
from utils import  network_model
#import ab_tf2
import param

batchSize=12
Epoch=250
#       import des data
#------------------------------------------------------------------------#
#import des données dans un dataset + normalisation


#dirMnist_128= "E:\\utt\\db\\MSTAR\\15_DEG_PNG\\128\\"
dirMstar=param.mstar_dir
dirMnist=param.mnist_128_dir

def our_generator():
    dir=dirMstar
    for subdir in os.listdir(dir):
      for filename in os.listdir(os.path.join(dir,subdir)):
        png_file=os.path.join(dir,subdir, filename)
        im = imageio.imread(png_file)
        im=(im - 127.5) /127.5
        im = im.reshape(im.shape[0], im.shape[1], 1)
        yield im

dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32)).shuffle(10000).batch(batchSize)

#------------------------------------------------------------------------#



#       reseau 
#------------------------------------------------------------------------#
generator = network_model.netG()
discriminator = network_model.netD()

generator_optimizer = tf.keras.optimizers.Adam(1e-6)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-6)

generator_loss=loss.wasserstein_generator_loss
discriminator_loss=loss.wasserstein_discriminator_loss

#------------------------------------------------------------------------#


#       metrics 
#------------------------------------------------------------------------#

discriminator_loss_M = tf.keras.metrics.Mean('d_loss', dtype=tf.float32)
discriminator_accuracy_M = tf.keras.metrics.BinaryCrossentropy(name='d_accuracy')
generator_loss_M = tf.keras.metrics.Mean('g_loss', dtype=tf.float32)

#accuracy calculer a la main contrairement a discriminator_accuracy_M qui utilise une fonction tf
#analyser que celle ci dans tensorboard
accuracy_Homemade= tf.keras.metrics.Mean('loss_home', dtype=tf.float32)


current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
gen_log_dir = './logs/gradient_tape/' + current_time + '/gen'
disc_log_dir = './logs/gradient_tape/' + current_time + '/disc'
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)
disc_summary_writer = tf.summary.create_file_writer(disc_log_dir)

img_log_dir = './logs/gradient_tape/' + current_time + '/img'
img_summary_writer = tf.summary.create_file_writer(img_log_dir)

#------------------------------------------------------------------------#



#       fonction train 
#------------------------------------------------------------------------#

def train_step(images,train_G):
    
    #train D 5 times
    for _ in range(5):
        noise = tf.random.normal([batchSize, 100])
        with tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            fake_output = discriminator(generated_images, training=True)

            real_output = discriminator(images, training=True)

            disc_loss = discriminator_loss(real_output, fake_output,images,generated_images,discriminator)
        
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



    #train G 
    noise = tf.random.normal([batchSize, 100])
    with tf.GradientTape() as gen_tape:
      generated_images = generator(noise, training=True)
      fake_output = discriminator(generated_images, training=True)
      gen_loss = generator_loss(fake_output)  


    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
 


    discriminator_loss_M(disc_loss)
    discriminator_accuracy_M(tf.ones_like(real_output),real_output)
    discriminator_accuracy_M(tf.zeros_like(fake_output),fake_output)
    generator_loss_M(gen_loss)

    accuracy_Homemade(loss.accuracy(real_output,fake_output))

#seed pour generer les img pour tensorboard
seed = (tf.random.normal([9, 100])) 

def train(dataset, epochs):
    train_G=False
    n=1
    for epoch in range(epochs):
        start = time.time()
        
        for image_batch in dataset:
            if image_batch.shape[0] == batchSize :
                train_step(image_batch,train_G)
            n=n+1
            train_G=(n%5==0)

        #summary img
        predictions = generator(seed, training=False)
        with img_summary_writer.as_default():
            tf.summary.image('img_generate', predictions, step=epoch,max_outputs=9)


        with gen_summary_writer.as_default():
            tf.summary.scalar('loss', generator_loss_M.result(), step=epoch)


        with disc_summary_writer.as_default():
            tf.summary.scalar('loss', discriminator_loss_M.result(), step=epoch)
            tf.summary.scalar('accuracy', discriminator_accuracy_M.result(), step=epoch)
            tf.summary.scalar('accuracy_h', accuracy_Homemade.result(), step=epoch)

        print ('Epoch {}  acc={} Time for epoch {} sec'
              .format(epoch + 1,accuracy_Homemade.result(), time.time()-start))

        # Reset metrics every epoch
        discriminator_loss_M.reset_states()
        generator_loss_M.reset_states()
        discriminator_accuracy_M.reset_states()
        accuracy_Homemade.reset_states()

#------------------------------------------------------------------------#


print( "*** START ***")
train(dataset,Epoch )