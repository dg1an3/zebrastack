from keras.layers import Input, Dense, Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Model
from keras import regularizers
from keras.datasets import mnist
from keras.datasets import cifar10

import numpy as np
use_cifar10 = False
if use_cifar10:
    (x_train, _), (x_test, _) = cifar10.load_data()
else:
    (x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))
print(x_train.shape)
print(x_test.shape)

if use_cifar10: 
    encoding_dim = int(3072*3) # N > M floats
else: 
    encoding_dim = 128 # 32 floats for sparse -> compression of factor 24.5, assuming input is 784 floats
input_img = Input(shape=(28,28,1)) # x_train.shape[1:])
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', activity_regularizer=regularizers.l1(1.0e-6), padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test,x_test))

decoded_imgs = autoencoder.predict(x_test)

from show_original_decoded import show_original_decoded
show_original_decoded(x_train, decoded_imgs, 28)
