from tensorflow.keras.layers import Dense, Input, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import LocallyConnected2D, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import ActivityRegularization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

sz = 128
latent_dim = 12
locally_connected_channels = 2
do_plot_model = False
cit_decimate = False
act_func = 'softplus' # or 'relu'
use_mse = False

def sampling(args):
    """
    Reparameterization trick by sampling fr an isotropic unit Gaussian.
    instead of sampling from Q(z|X), sample eps = N(0,I) 
        then z = z_mean + sqrt(var)*eps    
    # Arguments
        args (tensor tuple): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """    
    z_mean, z_log_var = args
    
    # from keras import backend as K
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def vae_loss(z_mean, z_log_var, y_true, y_pred):
    """
    Compute VAE loss, using either mse or crossentropy.
    # Arguments
        z_mean: mean of Q(z|X)
        z_log_var: log variance of Q(z|X)
        y_true, y_pred: truth and predicated values
    # Returns
        loss value
    """
    from tensorflow.keras.losses import mse, binary_crossentropy
    # from keras import backend as K
    img_pixels = 1.0 # sz * sz * 100.0
    if use_mse:
        match_loss = mse(K.flatten(y_true), K.flatten(y_pred)) * img_pixels
    else:
        match_loss = img_pixels * \
            binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
            # binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return match_loss + (1e-6 * kl_loss)

def create_encoder():
    """
    """
    # create encoder side
    retina = Input(shape=(sz,sz,1), name='retina_{}'.format(sz))

    v1_conv2d = Conv2D(32, (5,5), name='v1_conv2d', activation=act_func, padding='same')(retina)
    v1_maxpool = MaxPooling2D((2,2), name='v1_maxpool', padding='same')(v1_conv2d)
    v1_dropout = SpatialDropout2D(0.1, name='v1_dropout')(v1_maxpool)

    v2_conv2d = Conv2D(64, (1,1), name='v2_conv2d', activation=act_func, padding='same')(v1_dropout)
    v2_maxpool = MaxPooling2D((2,2), name='v2_maxpool', padding='same')(v2_conv2d)

    v4_conv2d = Conv2D(64, (3,3), name='v4_conv2d', activation=act_func, padding='same')(v2_maxpool)
    v4_maxpool = MaxPooling2D((2,2), name='v4_maxpool', padding='same')(v4_conv2d)

    pit_conv2d = Conv2D(64, (1,1), name='pit_conv2d', activation=act_func, padding='same')(v4_maxpool)
    pit_maxpool = MaxPooling2D((2,2), name='pit_maxpool', padding='same')(pit_conv2d)

    cit_conv2d = Conv2D(64, (3,3), name='cit_conv2d', activation=act_func, padding='same')(pit_maxpool)
    if cit_decimate:
        cit_maxpool = MaxPooling2D((2,2), name='cit_maxpool', padding='same')(cit_conv2d)
        cit_out = cit_maxpool
    else:
        cit_out = cit_conv2d

    ait_local = LocallyConnected2D(locally_connected_channels, (3,3), 
                                   name='ait_local', activation=act_func)(cit_out)
    # ait_padding = ZeroPadding2D(padding=(1,1), name='ait_padding')(ait_local)
    # x = MaxPooling2D((2,2), padding='same', name='ait_maxpool')(ait_padding)

    ait_regular = ActivityRegularization(l1=0.0e-4, l2=0.0e-4, name='ait_regular')(ait_local)

    # shape info needed to build decoder model
    shape = K.int_shape(ait_regular)
    # print(shape)

    # generate latent vector Q(z|X)
    pulvinar_flatten = Flatten(name='pulvinar_flatten')(ait_regular)
    pulvinar_dense = Dense(latent_dim, activation=act_func, name='pulvinar_dense')(pulvinar_flatten)
    z_mean = Dense(latent_dim, name='z_mean')(pulvinar_dense)
    z_log_var = Dense(latent_dim, name='z_log_var')(pulvinar_dense)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(retina, [z_mean, z_log_var, z], name='v1_to_pulvinar_encoder')
    encoder.summary()
    
    if do_plot_model: 
        plot_model(encoder, to_file='data\{}.png'.format(encoder.name), show_shapes=True)

    return retina, encoder, shape, [z_mean, z_log_var, z]

def create_decoder(shape):
    """
    """
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    pulvinar_dense_back = Dense(shape[1] * shape[2] * shape[3], name='pulvinar_dense_back', 
                                activation=act_func)(latent_inputs)
    pulvinar_antiflatten = Reshape((shape[1], shape[2], shape[3]), name='pulvinar_antiflatten')(pulvinar_dense_back)

    ait_padding_back = ZeroPadding2D(padding=(1,1), name='ait_padding_back')(pulvinar_antiflatten)
    ait_local_back = LocallyConnected2D(locally_connected_channels, (3,3), name='ait_local_back', 
                                        activation=act_func)(ait_padding_back)

    cit_padding_back = ZeroPadding2D(padding=(1,1), name='cit_padding_back')(ait_local_back)
    if cit_decimate:
        ait_upsample_back = UpSampling2D((2,2), name='ait_upsample_back')(cit_padding_back)
        ait_out_back = ait_upsample_back
    else:
        ait_out_back = cit_padding_back

    cit_conv2d_trans = Conv2DTranspose(64, (3,3), name='cit_conv2d_trans', 
                                       activation=act_func, padding='same')(ait_out_back)
    cit_upsample_back = UpSampling2D((2,2), name='cit_upsample_back')(cit_conv2d_trans)

    pit_conv2d_trans = Conv2DTranspose(64, (1,1), name='pit_conv2d_trans', 
                                       activation=act_func, padding='same')(cit_upsample_back)
    pit_upsample_back = UpSampling2D((2,2), name='pit_upsample_back')(pit_conv2d_trans)

    v4_conv2d_trans = Conv2DTranspose(64, (3,3), name='v4_conv2d_trans', 
                                      activation=act_func, padding='same')(pit_upsample_back)
    v4_upsample_back = UpSampling2D((2,2), name='v4_upsample_back')(v4_conv2d_trans)

    v2_conv2d_trans = Conv2DTranspose(32, (1,1), name='v2_conv2d_trans', 
                                      activation=act_func, padding='same')(v4_upsample_back)
    v2_upsample_back = UpSampling2D((2,2), name='v2_upsample_back')(v2_conv2d_trans)

    v1_conv2d_5x5_back = Conv2DTranspose(1, (5,5), name='v1_conv2d_5x5_back', 
                                activation='sigmoid', padding='same')(v2_upsample_back)
    decoder = Model(latent_inputs, v1_conv2d_5x5_back, name='pulvinar_to_v1_decoder')
    decoder.summary()
    if do_plot_model: 
        plot_model(decoder, to_file='data\{}.png'.format(decoder.name), show_shapes=True)
        
    return decoder

def create_autoencoder(retina, encoder, prob_model, decoder):
    """
    """
    autoencoder_output = decoder(encoder(retina)[2])
    autoencoder = Model(retina, autoencoder_output, name='v1_to_pulvinar_vae')

    # now compile with the optimizer VAE loss function
    optimizer = 'adadelta'
    [z_mean, z_log_var, z] = prob_model
    autoencoder.compile(optimizer=optimizer, 
                        loss=lambda y_true, y_pred: vae_loss(z_mean, z_log_var, y_true, y_pred))
    autoencoder.summary()
    if do_plot_model: 
        plot_model(autoencoder, to_file='data\{}.png'.format(autoencoder.name), show_shapes=True)
        
    return autoencoder