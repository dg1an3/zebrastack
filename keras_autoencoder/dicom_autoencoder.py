import random
import numpy as np
from read_imageset import read_imageset_arrays
from build_autoencoder import build_autoencoder
from show_original_decoded import show_original_decoded

if __name__ == '__main__':

    sz = 60
    dataset_name = 'SPIE-AAPM'
    dataset_name = 'LIDC-IDRI'

    x_train = read_imageset_arrays(dataset_name, sz, 1.0)
    x_test = np.array(random.sample(list(x_train), int(len(x_train)/10)))

    autoencoder, encode_only = build_autoencoder(sz, 'adadelta', 'mean_squared_error')
    autoencoder.fit(x_train, x_train, 
                    epochs=100, batch_size=256, 
                    shuffle=True, validation_data=(x_test,x_test))

    decoded_imgs = autoencoder.predict(x_test)
    encode_only_imgs = encode_only.predict(x_test)
    show_original_decoded(x_test, decoded_imgs, 60)
