### Frechet Inception Distance (FID) ###

import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from tqdm import tqdm
import math
from scipy import linalg



inception_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling='avg')


def compute_embedding(dataloader, count):
	image_embeddings=[]

	for _ in tqdm(range(count)):
		images=next(iter(dataloader))
		embeddings = inception_model.predict(images)

		image_embeddings.extend(embeddings)

	return np.array(image_embeddings)




def calculate_Fid(real_embeddings, generated_embeddings):

	mu1,sigma1=real_embeddings.mean(axis=0),np.cov(real_embeddings,rowvar=False)
	mu2,sigma2=generated_embeddings.mean(axis=0),np.cov(generated_embeddings,rowvar=False)



	#matric de covariance
	#calculate sqrt of product between cov
	covmean =linalg.sqrtm(sigma1.dot(sigma2))

	if np.iscomplexobj(covmean):
		covmean = covmean.real()  #correct imginnary number from sqrt


	fid = np.sum((mu1-mu2)**2.0) + np.trace(sigma1+sigma2 - 2.0*covmean)

	return fid



def resize_and_preprocessReal(image):
    # convert grayscale image to rgb
    #image = tf.expand_dims(image, axis=-1)
    image = tf.image.grayscale_to_rgb(image)

    # resize the image to the expected input size for inception model
    image = tf.image.resize(image, (229,229), method='nearest')

    # preprocess image requied by inception model
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image)

    return image

def resize_and_preprocessGenerated(image):
    # convert grayscale image to rgb
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.grayscale_to_rgb(image)

    # resize the image to the expected input size for inception model
    image = tf.image.resize(image, (229,229), method='nearest')

    # preprocess image requied by inception model
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.inception_v3.preprocess_input(image)

    return image


def fid(realloader,genloader,batchSize=256):

	count = math.ceil(1000/batchSize)

	#convert to rgb for inceptionv3
	realloader=(realloader.map(resize_and_preprocessReal))
	genloader=(genloader.map(resize_and_preprocessGenerated))

	real_image_embeddings=compute_embedding(realloader, count)

	generated_image_embeddings = compute_embedding(genloader, count)

	fid = calculate_Fid(real_image_embeddings,generated_image_embeddings)

	return fid