import random
import numpy as np
from read_imageset import read_imageset_arrays
from build_autoencoder import build_autoencoder
from show_original_decoded import show_original_decoded

if __name__ == '__main__':

    sz = 128
    dataset_name = 'SPIE-AAPM'
    # dataset_name = 'LIDC-IDRI'

    x_train = read_imageset_arrays(dataset_name, sz, 0.1)
    x_test = np.array(random.sample(list(x_train), int(len(x_train)/10)))

    autoencoder, encode_only, decode_only = build_autoencoder(sz, 'adadelta', 'mean_squared_error')
    autoencoder.fit(x_train, x_train, 
                    epochs=80, batch_size=256, 
                    shuffle=True, validation_data=(x_test,x_test))

    # decoded_imgs = autoencoder.predict(x_test)


    encode_only_imgs = encode_only.predict(x_test)
    for n in range(10):
        print("shape of encoded = ", encode_only_imgs[n].shape)
        hist, bins = np.histogram(encode_only_imgs[n])
        print(hist)
        print(bins)

    # add random values to decoded
    perturb_vectors = np.random.standard_normal(size=encode_only_imgs.shape)
    perturb_vectors = np.multiply(perturb_vectors, 0.8)
    encode_only_imgs = np.add(encode_only_imgs, perturb_vectors)

    decoded_imgs = decode_only.predict(encode_only_imgs)
    show_original_decoded(x_test, decoded_imgs, sz)
