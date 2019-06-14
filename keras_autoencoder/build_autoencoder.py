from keras.models import Model
from keras.layers import Input, SpatialDropout2D, Conv2D, LocallyConnected2D, ZeroPadding2D, MaxPooling2D, UpSampling2D, ActivityRegularization
from keras import regularizers

def build_autoencoder(sz, optimizer, loss):
    input_img = Input(shape=(sz,sz,1))
    print(input_img.shape)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    print(x.shape)
    x = MaxPooling2D((2,2), padding='same')(x)
    print(x.shape)
    x = SpatialDropout2D(0.1)(x)
    print(x.shape)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    print(x.shape)
    x = MaxPooling2D((2,2), padding='same')(x)
    print(x.shape)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    print(x.shape)
    x = MaxPooling2D((2,2), padding='same')(x)
    print(x.shape)
    x = LocallyConnected2D(16, (3,3))(x)
    print(x.shape)
    x = ZeroPadding2D(padding=(1,1))(x)
    print(x.shape)
    x = MaxPooling2D((2,2), padding='same')(x)
    print(x.shape)
    encoded = ActivityRegularization(l1=5.0e-6,l2=1.0e-6)(x)

    # TODO: add threshold layer for sparsity test

    x = LocallyConnected2D(16, (3,3))(x)
    print(x.shape)
    x = ZeroPadding2D(padding=(1,1))(x)
    print(x.shape)
    x = UpSampling2D((2,2))(x)
    print(x.shape)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    print(x.shape)
    x = UpSampling2D((2,2))(x)
    print(x.shape)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    print(x.shape)
    x = UpSampling2D((2,2))(x)
    print(x.shape)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    print(x.shape)
    x = UpSampling2D((2,2))(x)
    print(x.shape)
    decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)
    print(decoded.shape)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    encode_only = Model(input_img, encoded)
    return autoencoder, encode_only

