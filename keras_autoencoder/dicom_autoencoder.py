import random
import numpy as np
from read_imageset import read_imageset_arrays
from build_autoencoder import build_autoencoder
from show_original_decoded import show_original_decoded

if __name__ == '__main__':

    sz = 128
    dataset_name = 'SPIE-AAPM'
    # dataset_name = 'LIDC-IDRI'

    x_train = read_imageset_arrays(dataset_name, sz, 1.0)
    x_test = np.array(random.sample(list(x_train), int(len(x_train)/10)))

    autoencoder, encode_only, decode_only = build_autoencoder(sz, 'adadelta', 'mean_squared_error')
    autoencoder.fit(x_train, x_train, 
                    epochs=100, batch_size=256, 
                    shuffle=True, validation_data=(x_test,x_test))

    # decoded_imgs = autoencoder.predict(x_test)


    encode_only_imgs = encode_only.predict(x_test)
    for n in range(10):
        print("shape of encoded = ", encode_only_imgs[2].shape)
        hist, bins = np.histogram(encode_only_imgs[2])
        print(hist)
        print(bins)

    # add random values to decoded
    perturb_vectors = np.random.standard_normal(size=encode_only_imgs[2].shape)
    perturb_vectors = np.multiply(perturb_vectors, 5.6)
    encode_only_imgs_z = np.add(encode_only_imgs[2], perturb_vectors)

    decoded_imgs = decode_only.predict(encode_only_imgs_z)
    show_original_decoded(x_test, decoded_imgs, sz)


    # add random values to decoded
    perturb_vectors = np.random.standard_normal(size=encode_only_imgs[2].shape)
    perturb_vectors = np.multiply(perturb_vectors, 3.8)
    encode_only_imgs_z = np.add(encode_only_imgs[2], perturb_vectors)

    decoded_imgs = decode_only.predict(encode_only_imgs_z)
    show_original_decoded(x_test, decoded_imgs, sz)


    # add random values to decoded
    perturb_vectors = np.random.standard_normal(size=encode_only_imgs[2].shape)
    perturb_vectors = np.multiply(perturb_vectors, 1.9)
    encode_only_imgs_z = np.add(encode_only_imgs[2], perturb_vectors)

    decoded_imgs = decode_only.predict(encode_only_imgs_z )
    show_original_decoded(x_test, decoded_imgs, sz)
