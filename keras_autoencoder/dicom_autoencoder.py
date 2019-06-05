import os, time, random
import pydicom
from pydicom.data import get_testdata_files
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.models import Model
from keras import regularizers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import save_img
import keras.regularizers


dataset_directory = "E:\\Data\\tcia\\SPIE-AAPM Lung CT Challenge"
# dataset_directory = "E:\\Data\\tcia\\LIDC-IDRI"
imagesets = []
datasets = [(path, files) for path, _, files in os.walk(dataset_directory) if len(files) > 0]
for path, files in datasets:
    dss = [pydicom.dcmread(path+ "\\"+ file) for file in files if file.endswith('dcm')]
    dss = [(float(ds.ImagePositionPatient[2]), ds) for ds in dss if hasattr(ds, 'ImagePositionPatient')]
    dss = random.sample(dss, int(len(dss)/10))
    dss.sort(key=lambda pair:pair[0])
    for z, ds in dss:
        print(ds.filename, z)
        x = ds.pixel_array
        x = x.astype('float32')
        slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
        x = np.multiply(x, slope)
        x = np.add(x, intercept)
        x = resize(x, (28,28))
        x = np.add(x, 300.0)
        x = np.divide(x, 400.0)
        x = x.clip(0.0, 1.0)
        imagesets.append((z,x))
x_train = np.array([x for _, x in imagesets])
x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = np.array(random.sample(list(x_train), int(len(x_train)/10)))

encoding_dim = 128 # 32 floats for sparse -> compression of factor 24.5, assuming input is 784 floats
input_img = Input(shape=(28,28,1)) # x_train.shape[1:])
x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), activation='relu', activity_regularizer=regularizers.l1(1.0e-8), padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test,x_test))

decoded_imgs = autoencoder.predict(x_test)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False) 

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show(block=True)
print('done')
