
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
    

    Dense1=layers.Dense(32*32*64, use_bias=False)(joined)
    BN1=layers.BatchNormalization()(Dense1)
    Relu1=layers.LeakyReLU()(BN1)

    Reshape=layers.Reshape((32, 32, 64))(Relu1)

    ConvT1=layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False) (Reshape)
    BN2=layers.BatchNormalization()(ConvT1)
    Relu2=layers.LeakyReLU()(BN2)

    ConvT2=layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False) (Reshape)
    BN3=layers.BatchNormalization()(ConvT2)
    Relu3=layers.LeakyReLU()(BN3)

    Output=layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh') (Relu3)
    
    return tf.keras.Model([z_input,y_input],Output)


#D model
def make_discriminator_conditional_model(nb_classes=10,img_shape=(128,128,3)):

    img_input=layers.Input(shape=img_shape)

    y_input=layers.Input(shape=(1,))
    y_embedding=layers.Embedding(nb_classes,np.prod(img_shape),input_length=1)(y_input)
    y_flatten=layers.Flatten()(y_embedding)
    y_reshape=layers.Reshape(img_shape)(y_flatten)

    concatenated=layers.Concatenate(axis=-1)([img_input,y_reshape])


    Conv1=layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=(128,128,4))(concatenated) 
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