import os, time, random
import pydicom
from pydicom.data import get_testdata_files
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import save_img
import keras.regularizers


from read_imageset import read_imageset_arrays
# x_train = read_imageset_arrays('SPIE-AAPM', 60)
x_train = read_imageset_arrays('LIDC-IDRI', 60)
x_test = np.array(random.sample(list(x_train), int(len(x_train)/10)))

encoding_dim = 128 # 32 floats for sparse -> compression of factor 24.5, assuming input is 784 floats
input_img = Input(shape=(60,60,1)) # x_train.shape[1:])
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', activity_regularizer=regularizers.l1(1.0e-6), padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_test,x_test))

decoded_imgs = autoencoder.predict(x_test)

# TODO: move this to display_imageset

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(60, 60))
    
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    
    plt.imshow(decoded_imgs[i].reshape(60, 60))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show(block=True)
print('done')
