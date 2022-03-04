import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10

from load_mstar import get_mstar_perso

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid



model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

train_img ,train_attribute ,test_img,test_attribute = get_mstar_perso()



numpy.random.shuffle(train_img)
numpy.random.shuffle(test_img)

print('Loaded', train_img.shape, test_img.shape)

# convert integer to floating point values
images1 = train_img.astype('float32')
images2 = test_img.astype('float32')

# resize images
train_img = scale_images(train_img, (299,299,3))
test_img = scale_images(test_img, (299,299,3))


print('Scaled', train_img.shape, test_img.shape)


# pre-process images
images1 = preprocess_input(train_img)
images2 = preprocess_input(test_img)


fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid)

#train_img = train_img.astype('float32')
#test_img = test_img.astype('float32')