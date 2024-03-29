""" model_vae_3stage module
encapsulates VAE model in a class
"""

import random

from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import Dense, Input, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import LocallyConnected2D, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import ActivityRegularization, Activation

from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

from utils import get_initial_weights
from layers import BilinearInterpolation

def pre_STN(input_shape=(60, 60, 1), sampling_size=(30, 30)):
    input_image = Input(shape=input_shape, name='input_img')
    locnet = MaxPooling2D(pool_size=(2, 2))(input_image)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = MaxPooling2D(pool_size=(2, 2))(locnet)
    locnet = Conv2D(20, (5, 5))(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(50)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_weights(50)
    locnet = Dense(6, weights=weights)(locnet)
    x = BilinearInterpolation(sampling_size)([input_image, locnet])
    return x, input_image

def build_encoded_layer(input_img, l1_l2=(0.0e-4, 0.0e-4), use_dropout=True):
    """Create encoded layer, prior to projection to latent space."""
    x = Conv2D(8, (3, 3), activation=relu, padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    if use_dropout:
        x = SpatialDropout2D(0.1)(x)
    x = Conv2D(16, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), activation=relu, padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = LocallyConnected2D(32, (3, 3))(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    l1, l2 = l1_l2
    encoded_layer = ActivityRegularization(l1=l1, l2=l2)(x)
    return encoded_layer

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_latent_encoder(encoded_layer, input_img, latent_dim=8, dump=False):
    """Create projection to latent vector Q(z|X)."""
    x = Flatten()(encoded_layer)
    x = Dense(32, activation=relu)(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(input_img, [z_mean, z_log_var, z], name='vae_encoder')
    if dump:
        encoder.summary()
        plot_model(encoder, to_file='dicom_encoder.png', show_shapes=True)

    # TODO: add threshold layer for sparsity test
    return encoder, z_mean, z_log_var

def build_decoder(size, encoded_shape, in_channels=1, latent_dim=8, dump=False):
    """Create decoder from latent space back to grid."""
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(encoded_shape[1] * encoded_shape[2] * encoded_shape[3], activation=relu)(latent_inputs)
    x = Reshape((encoded_shape[1], encoded_shape[2], encoded_shape[3]))(x)

    x = LocallyConnected2D(32, (3,3))(x)
    x = ZeroPadding2D(padding=(1,1))(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3,3), activation=relu, padding='same')(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(16, (3,3), activation=relu, padding='same')(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(16, (3,3), activation=relu, padding='same')(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(8, (3,3), activation=relu, padding='same')(x)

    x = UpSampling2D((2,2))(x)
    # can't use Activation(sigmoid)(x) because we need a conv layer to reduce 4-channels to 1-channel
    decoded_layer = Conv2D(1, (3,3), activation=relu, padding='same')(x)
    decoder = Model(latent_inputs, decoded_layer, name='vae_decoder')
    if dump:
        decoder.summary()
        plot_model(decoder, to_file='dicom_decoder.png', show_shapes=True)

    return decoder

def build_autoencoder(encoder, decoder, input_img, loss, optimizer='ada', dump=False):
    """builds an autoencoder from an encoder/decoder pair."""
    autoencoder_output = decoder(encoder(input_img)[2])
    autoencoder = Model(input_img, autoencoder_output, name='vae')

    autoencoder.compile(optimizer=optimizer, loss=loss)
    if dump:
        autoencoder.summary()
        plot_model(autoencoder, to_file='dicom_autoencoder.png', show_shapes=True)
    return autoencoder

class ModelVae3Stage:
    """
    Wraps all three parts of the VAE model: encoder, decoder, and vae.
    * Arguments: else
    """

    def __init__(self, size=64, in_channels=1, latent_dim=8, use_kldiv=True):
        self.size = size

        xformed_img, input_img = pre_STN(input_shape=(size,size,in_channels), sampling_size=(size//2,size//2))
        # xformed_img = Reshape((size,size,in_channels))(xformed_img)

        # TODO: try to read stored model, if available
        encoded_layer = build_encoded_layer(xformed_img)
        self.encoder, self.z_mean, self.z_log_var = \
            build_latent_encoder(encoded_layer, input_img, latent_dim=latent_dim)
        
        # shape info needed to build decoder model
        encoded_shape = K.int_shape(encoded_layer)
        self.decoder = \
            build_decoder(size, encoded_shape, in_channels, latent_dim=latent_dim)

        bind_loss = lambda y_true, y_pred: self.vae_loss(y_true, y_pred)
        self.vae = \
            build_autoencoder(self.encoder, self.decoder, xformed_img,  loss=bind_loss, optimizer='adadelta')

    def __str__(self):
        # output as yaml
        return 'Model3StageVae'

    def vae_loss(self, y_true, y_pred):
        """Compute VAE loss, using either mse or crossentropy."""
        img_pixels = self.size * self.size
        use_mse = True
        if use_mse:
            match_loss = mse(K.flatten(y_true), K.flatten(y_pred)) * img_pixels
        else:
            match_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred)) * img_pixels
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(match_loss + kl_loss)

    def train(self, x_train, epochs=1000):
        x_test = np.array(random.sample(list(x_train), int(len(x_train)/10))) 
        self.vae.train(x_train, x_train, 
                       epochs=epochs, batch_size=256, 
                       shuffle=True, validation_data=(x_test, x_test))
        # TODO: write out saved model after each training

    def predict(self, x_test):
        return self.vae.predict(x_test)