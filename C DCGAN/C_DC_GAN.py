print("---start---")
print("C-DCGAN")


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

from IPython import display



print(" import\n\n")
#------------------------------------------------------------------------#
print(" param")


BATCH_SIZE = 256
EPOCHS = 5
noise_dim = 100
num_examples_to_generate = 16

print(" param\n\n")
#------------------------------------------------------------------------#
print(" dataset")



#import
(x_train,y_train),(_,_)=tf.keras.datasets.mnist.load_data()

#reshape and norm
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train= ( x_train - 127.5 ) / 127.5

#suffle
train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(x_train.shape[0]).batch(BATCH_SIZE)



print(" dataset\n\n")
#------------------------------------------------------------------------#
print(" def model")


#G model
def make_generator_model(nb_classes=10,z_dim=100):
    

    #latent input of size 100*?
    z_input=layers.Input(shape=(z_dim,))

    #label input of size 1*?
    y_input=layers.Input(shape=(1,))
    y_embedding=layers.Embedding(nb_classes,z_dim,input_length=1)(y_input)
    y_flatten=layers.Flatten()(y_embedding)

    joined=layers.multiply([z_input,y_flatten])
    
    
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

    img=model(joined)

    return tf.keras.Model([z_input,y_input],img)


#D model
def make_discriminator_model(nb_classes=10,img_shape=(28,28,1)):

    img_input=layers.Input(shape=img_shape)

    y_input=layers.Input(shape=(1,))
    y_embedding=layers.Embedding(nb_classes,np.prod(img_shape),input_length=1)(y_input)
    y_flatten=layers.Flatten()(y_embedding)
    y_reshape=layers.Reshape(img_shape)(y_flatten)

    concatenated=layers.Concatenate(axis=-1)([img_input,y_reshape])


    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 2]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    #prediction = model(concatenated)
    #return Model([img, label], prediction)
    
    return tf.keras.Model([img_input, y_input],model(concatenated))

generator = make_generator_model()
print(generator.summary())
discriminator = make_discriminator_model()
print(discriminator.summary())


print(" def model\n\n")
#------------------------------------------------------------------------#
print(" def loss")



cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)



print(" def loss\n\n")
#------------------------------------------------------------------------#
print(" def train")




#@tf.function
def train_step(images):
    print("--->train_step")
    noise = tf.random.normal([images[-1].numpy().size, noise_dim])
    print("   train_step 1")
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      print("   train_step 2")
      #images[1].numpy() = label of the images
      generated_images = generator((noise,images[1]), training=True)
      print("   train_step 3")
      real_output = discriminator((images[0],images[1]), training=True)
      fake_output = discriminator((generated_images,images[1]), training=True)
      print("   train_step 4")
      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)
    print("   train_step 5")
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    print("   train_step 6")
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print("   train_step 7")
    print("<---train_step")
    return (gen_loss,disc_loss)

    
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])    
    
def train(dataset, epochs):
    print("--->train")
    gen_loss,disc_loss =0,0
    for epoch in range(epochs):
        start = time.time()
        print("   train1")
        for image_batch in dataset:
            print("   train2")
            (gen_loss,disc_loss) = train_step(image_batch)
        print("   train3")
        # Produce images for the GIF
        display.clear_output(wait=True)
        print("   train4")
        generate_and_save_images(generator,epoch + 1,seed)

        print("   train5")
        print ('Epoch {} LossG = {}   LossD={}  Time for epoch {} sec'
              .format(epoch + 1,gen_loss,disc_loss, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    print("   train6")
    generate_and_save_images(generator,epochs,seed)
  
  
def generate_and_save_images(model, epoch, test_input):
  print("--->generate_and_save_images")
  predictions = model(([test_input],np.array([0,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6])), training=False)
  print("   generate_and_save_images2")
  fig = plt.figure(figsize=(4, 4))
  print("   generate_and_save_images3")
  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
  print("   generate_and_save_images4")
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  print("<---generate_and_save_images")



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

