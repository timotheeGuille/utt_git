import param
import tensorflow as tf
import tensorflow_datasets as tfds



def importMnist():

    (x_train,y_train),(_,_)=tf.keras.datasets.mnist.load_data()

    #reshape and norm
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_train= ( x_train - 127.5 ) / 127.5

    #suffle
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(x_train.shape[0]).batch(param.BATCH_SIZE)

    return train_dataset

def importMSTAR():
       
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
    param.mstar_dir,
    labels="inferred",
    label_mode="int",  # categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    color_mode="grayscale",
    batch_size=param.BATCH_SIZE,
    image_size=(128, 128),
    shuffle=True,
    seed=123,
    )

    return dataset 

def importMNIST128():
       
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
    param.mnist_128_dir,
    labels="inferred",
    label_mode="int",  # categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    color_mode="grayscale",
    batch_size=param.BATCH_SIZE,
    image_size=(128, 128),
    shuffle=True,
    seed=123,
    )

    return dataset 

def importMNIST28_local():
       
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
    param.mnist_28_dir,
    labels="inferred",
    label_mode="int",  # categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    color_mode="grayscale",
    batch_size=param.BATCH_SIZE,
    image_size=(28, 28),
    shuffle=True,
    seed=123,
    )

    return dataset 