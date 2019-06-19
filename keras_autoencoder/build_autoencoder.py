from keras.models import Model
from keras.layers import Input, SpatialDropout2D, Conv2D, Conv2DTranspose, LocallyConnected2D, ZeroPadding2D, MaxPooling2D, UpSampling2D, ActivityRegularization
from keras.utils import plot_model
from keras import regularizers

def build_autoencoder(sz, optimizer, loss):
    input_img = Input(shape=(sz,sz,1))
    x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = SpatialDropout2D(0.1)(x)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = LocallyConnected2D(16, (3,3))(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    encoded = ActivityRegularization(l1=0.0e-4,l2=0.0e-4)(x)

    # TODO: add threshold layer for sparsity test

    x = LocallyConnected2D(16, (3,3))(encoded)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(16, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(16, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(32, (3,3), activation='relu', padding='same')(x)
    x = UpSampling2D((2,2))(x)
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.summary()
    plot_model(autoencoder, to_file='data\dicom_autoencoder.png', show_shapes=True)

    encode_only = Model(input_img, encoded)

    encoded_input = Input(batch_shape=(None,encoded.shape[1].value,encoded.shape[2].value,encoded.shape[3].value))
    x = autoencoder.layers[-10](encoded_input)
    x = autoencoder.layers[-9](x)
    x = autoencoder.layers[-8](x)
    x = autoencoder.layers[-7](x)
    x = autoencoder.layers[-6](x)
    x = autoencoder.layers[-5](x)
    x = autoencoder.layers[-4](x)
    x = autoencoder.layers[-3](x)
    x = autoencoder.layers[-2](x)
    x = autoencoder.layers[-1](x)
    decode_only = Model(encoded_input, x)

    return autoencoder, encode_only, decode_only


