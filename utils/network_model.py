
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np




def make_generator_conditional_model(nb_classes=10,z_dim=100):
    

    #latent input of size 100*?
    z_input=layers.Input(shape=(z_dim,))

    #label input of size 1*?
    y_input=layers.Input(shape=(1,))
    y_embedding=layers.Embedding(nb_classes,z_dim,input_length=1)(y_input)
    y_flatten=layers.Flatten()(y_embedding)

    joined=layers.multiply([z_input,y_flatten])
    

    Dense1=layers.Dense(16*16*64, use_bias=False)(joined)
    BN1=layers.BatchNormalization()(Dense1)
    Relu1=layers.LeakyReLU()(BN1)

    Reshape=layers.Reshape((16, 16, 64))(Relu1)

    ConvT1=layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False) (Reshape)
    BN2=layers.BatchNormalization()(ConvT1)
    Relu2=layers.LeakyReLU()(BN2)

    ConvT2=layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False) (Relu2)
    BN3=layers.BatchNormalization()(ConvT2)
    Relu3=layers.LeakyReLU()(BN3)

    ConvT3=layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False) (Relu3)
    BN4=layers.BatchNormalization()(ConvT3)
    Relu4=layers.LeakyReLU()(BN4)

    Output=layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh') (Relu4)
    
    return tf.keras.Model([z_input,y_input],Output)


#D model
def make_discriminator_conditional_model(nb_classes=10,img_shape=(128,128,1)):

    img_input=layers.Input(shape=img_shape)

    y_input=layers.Input(shape=(1,))
    y_embedding=layers.Embedding(nb_classes,np.prod(img_shape),input_length=1)(y_input)
    y_flatten=layers.Flatten()(y_embedding)
    y_reshape=layers.Reshape(img_shape)(y_flatten)

    concatenated=layers.Concatenate(axis=-1)([img_input,y_reshape])


    Conv1=layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=(128,128,2))(concatenated) 
    BN1=layers.BatchNormalization()(Conv1)
    Relu1=layers.LeakyReLU()(BN1)

    Conv2=layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(Relu1) 
    BN2=layers.BatchNormalization()(Conv2)
    Relu2=layers.LeakyReLU()(BN2)

    Conv3=layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(Relu2) 
    BN3=layers.BatchNormalization()(Conv3)
    Relu3=layers.LeakyReLU()(BN3)

    Flat1=layers.Flatten()(Relu3)
    Output=layers.Dense(1)(Flat1)
    
    return tf.keras.Model([img_input, y_input],Output)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*64, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 64)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[128, 128, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model