import os
import pydicom
import numpy as np

def get_imageset_id(ds):
    return ds.PatientID + '_' + ds.AcquisitionDate

def read_slice(ds, sz):
    if not(hasattr(ds, 'ImagePositionPatient')):
        return None
    print(ds.filename, ds.ImagePositionPatient[2])
    slice_array = ds.pixel_array
    slice_array = slice_array.astype('float32')
    slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
    slice_array = np.multiply(slice_array, slope)
    slice_array = np.add(slice_array, intercept)

    slice_array = resize(slice_array, (sz,sz))

    slice_array = np.add(slice_array, 200.0)
    slice_array = np.divide(slice_array, 400.0)
    slice_array = slice_array.clip(0.0, 1.0)
    return slice_array

def process_imageset(path_name, dataset_name, sz):
    for path, _, files in os.walk(path_name):
        if len(files) == 0:
            continue
        imageset_id = None
        images = []
        for file in files:
            if not(file.endswith('dcm')):
                continue
            ds = pydicom.dcmread('\\'.join(path, file))
            imageset_id = get_imageset_id(ds)
            slice_array = read_slice(ds, sz)
            images.append(x)
        images = np.array(images)
        images = images.reshape((len(images), sz, sz, 1)) 
        np.save('\\'.join(['.', 'data', dataset_name, "{}x{}".format(sz,sz), imageset_id]), images)
        return imageset_id, images

def read_imageset_arrays(dataset_name, sz):
    slice_arrays = None
    path_name = '\\'.join(['.', 'data', dataset_name, "{}x{}".format(sz,sz)])
    for path, _, files in os.walk(path_name):
        if len(files) == 0:
            continue
        for file in files:
            if not(file.endswith('npy')):
                continue
            slice_array = np.load('\\'.join([path, file]))
            if (slice_arrays is None):
                slice_arrays = slice_array
            else:
                slice_arrays = np.append(slice_arrays, slice_array, axis=0)
    return slice_arrays
            