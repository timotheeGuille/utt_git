
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers

from tensorflow.keras import Model


(x_train,y_train),(_,_)=tf.keras.datasets.mnist.load_data()

BATCH_SIZE=256


#cut size to avoid size bigger than batchsize
size_cut=x_train.shape[0]-x_train.shape[0]%BATCH_SIZE

x_train=x_train[0:size_cut,:,:]
y_train=y_train[0:size_cut]

#reshape and norm
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train= ( x_train - 127.5 ) / 127.5

#suffle
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(x_train.shape[0]).batch(BATCH_SIZE)



from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape = (28, 28, 1), # Shape of our images
                                include_top = False, # Leave out the last fully connected layer
                                weights = 'imagenet')


for layer in pre_trained_model.layers:
  layer.trainable = False


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.959):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True


from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(pre_trained_model.output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])