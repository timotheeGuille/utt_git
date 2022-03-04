import imageio
import os

import numpy as np


def normalize(image):
   return (image/127.5)-1.0

label_to_int={"BRDM_2":0,
                "BTR_60":1,
                "BTR70":2,
                "SN_132":3,
                "SN_812":4,
                "SN_9563":5,
                "SN_9566":6,
                "SN_C21":7,
                "SN_S7":8}

def our_generator(dir):
    
    for subdir in os.listdir(dir):
      label=label_to_int[subdir]    
      for filename in os.listdir(os.path.join(dir,subdir)):
        png_file=os.path.join(dir,subdir, filename)
        img = imageio.imread(png_file).astype('float32')
        img=img.reshape((128,128,1))
        img=(img - 127.5) /127.5
        img = img.reshape(img.shape[0], img.shape[1], 1)
        yield img, label




def get_mstar_perso():

    db_train=list(our_generator("E:\\utt\\db\\MSTAR\\15_DEG_PNG\\128\\"))

    train_img=[]
    train_attribute=[]
    for data in db_train:
        img = np.array(data[0])
        train_img.append(img)
        

        label = np.zeros((9))  
        label[data[1]] = 1     
        train_attribute.append(label)
        
    train_img = np.asarray(train_img)
    train_attribute = np.asarray(train_attribute)
    

    db_test=list(our_generator("E:\\utt\\db\\MSTAR\\17_DEG_PNG\\128\\"))

    test_img=[]
    test_attribute=[]
    for data in db_test:
        img = np.array(data[0])
        test_img.append(img)
        

        label = np.zeros((9))  
        label[data[1]] = 1     
        test_attribute.append(label)
        
    test_img = np.asarray(train_img)
    test_attribute = np.asarray(train_attribute)
    return train_img ,train_attribute ,test_img,test_attribute
