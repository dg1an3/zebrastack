from keras.layers import Input, Dense
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
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

if use_cifar10: 
    encoding_dim = int(3072*3) # N > M floats
else: 
    encoding_dim = 128 # 32 floats for sparse -> compression of factor 24.5, assuming input is 784 floats
input_img = Input(shape=(784,)) # x_train.shape[1:])
encoded = Dense(int(encoding_dim), activation='relu', activity_regularizer=regularizers.l1(0.000001))(input_img)
encoded = Dense(int(encoding_dim/2), activation='relu')(encoded)
encoded = Dense(int(encoding_dim/4), activation='relu')(encoded)

decoded = Dense(int(encoding_dim/2), activation='relu')(encoded)
decoded = Dense(int(encoding_dim), activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

# create separate decoder
encoded_input = Input(shape=(encoding_dim/4,))
decoder_layer = autoencoder.layers[-3](encoded_input)
decoder_layer = autoencoder.layers[-2](decoder_layer)
decoder_layer = autoencoder.layers[-1](decoder_layer)
decoder = Model(encoded_input, decoder_layer)

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(x_train, x_train, epochs=100, batch_size=256, shuffle=True, validation_data=(x_test,x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    if use_cifar10: plt.imshow(x_test[i].reshape(32, 32, 3))
    else: plt.imshow(x_test[i].reshape(28, 28))
    
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    
    if use_cifar10: plt.imshow(decoded_imgs[i].reshape(32, 32, 3))
    else: plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
print('done')