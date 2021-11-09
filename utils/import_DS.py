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
    
    builder = tfds.ImageFolder("/home/tim/Documents/utt/db/MSTAR/15_DEG_PNG/")
    print(builder.info)

    #ds = builder.as_dataset(split='128',shuffle_files=True)
    #tfds.show_examples(ds,builder.info)
    ds = builder.as_dataset(split='128',batch_size=param.BATCH_SIZE,shuffle_files=True, as_supervised=True)
    


    return ds