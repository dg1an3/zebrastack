import os, time, random
import pydicom
from pydicom.data import get_testdata_files
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.preprocessing.image import save_img

import numpy as np

if False:
    filename = get_testdata_files("rtplan.dcm")[0]
    ds = pydicom.dcmread(filename)
    print(ds.PatientName)
    print(ds.dir("setup"))
    print(ds.PatientSetupSequence[0])

if False:
    mr_filename = get_testdata_files("MR_small.dcm")[0]
    mr_ds = pydicom.dcmread(mr_filename)
    print(mr_ds.pixel_array)

if False:
    filename = get_testdata_files("CT_small.dcm")[0]
    ds = pydicom.dcmread(filename)
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 

if False:
    filename = "e:\\Data\\tcia\\SPIE-AAPM Lung CT Challenge\\CT-Training-BE001\\01-03-2007-16904-CT INFUSED CHEST-143.1\\4-HIGH RES-47.17\\000000.dcm"
    ds = pydicom.dcmread(filename)
    print(ds)
    plt.imshow(ds.pixel_array[::8,::8], cmap=plt.cm.bone) 
    plt.show()

if True:
    dataset_directory = "E:\\Data\\tcia\\SPIE-AAPM Lung CT Challenge"
    # dataset_directory = "E:\\Data\\tcia\\LIDC-IDRI"
    imagesets = []
    datasets = [(path, files) for path, _, files in os.walk(dataset_directory) if len(files) > 0]
    for path, files in datasets:
        dss = [pydicom.dcmread(path+ "\\"+ file) for file in files if file.endswith('dcm') and random.random() > 0.99]
        dss = [(float(ds.ImagePositionPatient[2]), ds) for ds in dss if hasattr(ds, 'ImagePositionPatient')]
        dss.sort(key=lambda pair:pair[0])
        for z, ds in dss:
            print(ds.filename, z)
            x = resize(ds.pixel_array, (28,28))
            slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
            x = np.multiply(x,slope)
            x = np.add(x, intercept)
            x = x.astype('float32') #/ 255.
            imagesets.append((z,x))
    x_train = np.array([x for _, x in imagesets])
    x_train = x_train.reshape((len(x_train), 28, 28, 1))
    for z, x_train in imagesets:
        # x_train = np.clip(x_train, 0.0, 1.0)
        plt.imshow(x_train, cmap=plt.cm.bone) 
        plt.pause(0.1)

print("done")
