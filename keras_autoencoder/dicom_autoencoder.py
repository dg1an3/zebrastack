import random
import numpy as np
from read_imageset import read_imageset_arrays
from build_autoencoder import build_autoencoder
from show_original_decoded import show_original_decoded

if __name__ == '__main__':

    sz = 60
    dataset_name = 'SPIE-AAPM' # 'LIDC-IDRI'

    x_train = read_imageset_arrays(dataset_name, sz)
    x_test = np.array(random.sample(list(x_train), int(len(x_train)/10)))

    autoencoder = build_autoencoder(sz, 'adadelta', 'mean_squared_error')
    autoencoder.fit(x_train, x_train, 
                    epochs=2, batch_size=256, 
                    shuffle=True, validation_data=(x_test,x_test))

    decoded_imgs = autoencoder.predict(x_test)
    show_original_decoded(x_train, decoded_imgs, 60)
