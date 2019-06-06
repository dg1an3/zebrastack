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

sz = 60

from read_imageset import read_imageset_arrays
x_train = read_imageset_arrays('SPIE-AAPM', sz)
# x_train = read_imageset_arrays('LIDC-IDRI', sz)
x_test = np.array(random.sample(list(x_train), int(len(x_train)/10)))

encoding_dim = 128 # 32 floats for sparse -> compression of factor 24.5, assuming input is 784 floats
input_img = Input(shape=(sz,sz,1)) # x_train.shape[1:])
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

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test,x_test))

from show_original_decoded import show_original_decoded
decoded_imgs = autoencoder.predict(x_test)
show_original_decoded(x_train, decoded_imgs, 60)
