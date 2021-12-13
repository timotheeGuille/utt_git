import tensorflow as tf
from tensorflow.keras import layers


def netG():
   z=100
   model = tf.keras.Sequential()
  
   model.add(layers.Dense(4*4*512,use_bias=False,input_shape=(z,)))

   model.add(layers.Reshape((4, 4, 512)))

   model.add(layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same'))
   model.add(layers.BatchNormalization())
   model.add(layers.ReLU())
   model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same'))
   model.add(layers.BatchNormalization())
   model.add(layers.ReLU())
   model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same'))
   model.add(layers.BatchNormalization())
   model.add(layers.ReLU())
   model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
   model.add(layers.BatchNormalization())
   model.add(layers.ReLU())
   model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),activation='tanh', padding='same'))



   return model

def netD():
   model = tf.keras.Sequential()

   model.add(layers.Input(shape=(128,128,1)))
    
    
   model.add(layers.Conv2D(64, (5, 5), strides=(2, 2),activation=tf.identity, padding='same',input_shape=[128, 128, 1]))
   model.add(layers.BatchNormalization())
   model.add(layers.ReLU())
   model.add(layers.Conv2D(128, (5, 5), strides=(2, 2),activation=tf.identity, padding='same'))
   model.add(layers.BatchNormalization())
   model.add(layers.ReLU())
   model.add(layers.Conv2D(256, (5, 5), strides=(2, 2),activation=tf.identity, padding='same'))
   model.add(layers.BatchNormalization())
   model.add(layers.ReLU())
   model.add(layers.Conv2D(512, (5, 5), strides=(2, 2),activation=tf.identity, padding='same'))
   model.add(layers.BatchNormalization())
   model.add(layers.ReLU())
   model.add(layers.Conv2D(1, (4, 4), strides=(1, 1),activation=tf.identity, padding='same'))
   model.add(layers.BatchNormalization())


   return model