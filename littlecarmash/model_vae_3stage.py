""" model_vae_3stage module
encapsulates VAE model in a class
"""

from keras.activations import relu, sigmoid
from keras.layers import Dense, Input, SpatialDropout2D
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import LocallyConnected2D, ZeroPadding2D
from keras.layers import MaxPooling2D, UpSampling2D
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import ActivityRegularization

from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

def build_encoded_layer(size, in_channels=1, latent_dim=8, l1_l2=(0.0e-4, 0.0e-4), use_dropout=False):
    """Create encoded layer, prior to projection to latent space."""
    input_img = Input(shape=(size, size, in_channels))

    x = Conv2D(32, (3, 3), activation=relu, padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    if use_dropout:
        x = SpatialDropout2D(0.1)(x)

    x = Conv2D(32, (3, 3), activation=relu, padding='same')(x)
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

def build_latent_encoder(encoded_layer, latent_dim=8, dump=False):
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
    x = Conv2DTranspose(32, (3,3), activation=relu, padding='same')(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(32, (3,3), activation=relu, padding='same')(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(32, (3,3), activation=relu, padding='same')(x)

    x = UpSampling2D((2,2))(x)
    decoded_layer = Conv2D(1, (3,3), activation=sigmoid, padding='same')(x)
    decoder = Model(latent_inputs, decoded_layer, name='vae_decoder')
    if dump:
        decoder.summary()
        plot_model(decoder, to_file='dicom_decoder.png', show_shapes=True)

    return decoder

def vae_loss(z_mean, z_log_var):
    """Compute the total VAE loss=binary loss + KLDiv.
    # Arguments
        ??? args (tensor): mean and log of variance of Q(z|X)
    # Returns
        ??? z (tensor): sampled latent vector    
    """
    return z_mean + K.exp(z_log_var)

def build_autoencoder(encoder, decoder, optimizer='ada', loss='mse', dump=False):
    """builds an autoencoder from an encoder/decoder pair."""
    autoencoder_output = decoder(encoder(input_img)[2])

    autoencoder = Model(input_img, autoencoder_output, name='vae')
    autoencoder.compile(optimizer=optimizer, loss=loss)
    if dump:
        autoencoder.summary()
        plot_model(autoencoder, to_file='dicom_autoencoder.png', show_shapes=True)
    return autoencoder

class ModelVae3Stage:
    """Wraps all three parts of the VAE model: encoder, decoder, and vae."""

    def __init__(self, in_channels=1, latent_dim=8, use_kldiv=False):
        self.encoded_layer = build_encoder(size, in_channels, latent_dim)
        self.encoder, z_mean, z_log_var = build_latent_encoder(self.encoded_layer)
        
        # shape info needed to build decoder model
        encoded_shape = K.int_shape(encoded_layer)
        self.decoder = build_decoder(size, encoded_shape, in_channels, latent_dim)
        if use_kldiv:
            loss = vae_loss(z_mean, z_log_var)
        else:
            loss = mse
        self.vae = build_autoencoder(self.encoder, self.decoder, optimizer='adadelta', loss=loss)

    def __str__(self):
        # output as yaml
        return 'Model3StageVae'

    def train(self, x_train, epochs):
        self.vae.train(x_train, x_train, epochs=epochs, batch_size=256)

    def predict(self, x_test):
        return self.vae.predict(x_test)