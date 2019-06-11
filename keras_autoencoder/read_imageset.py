import os, random
import pydicom
import numpy as np
from skimage.transform import resize

def get_imageset_npy_fname(ds):
    """
    returns the npy filename for the given DICOM instance
    """
    return ds.PatientID + '_' + ds.AcquisitionDate

def get_dataset_path(dataset_name, sz, npy_fname = None):
    """
    returns the path to the given dataset, and optionally the npy filename
    """
    parts = ['.', 'data', dataset_name, "{}x{}".format(sz,sz)]
    if not(npy_fname is None):
        parts.append(npy_fname)
    return '\\'.join(parts)

def read_slice(full_path):
    """
    reads a DICOM slice given the file full path
    """
    if not(full_path.endswith('dcm')):
        return None
    ds = pydicom.dcmread(full_path)
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
    return ds, slice_array

def import_dataset(path_name, dataset_name, sz):
    """
    import_dataset('e:\\Data\\tcia\\LIDC-IDRI', 'LIDC', 60)
    """
    for path, _, files in os.walk(path_name):
        if len(files) == 0:
            continue
        images, ds = [], None
        for file in files:
            ds, slice_array = read_slice('\\'.join([file, path]))
            images.append(slice_array)
        if (len(images) > 0):
            images = np.array(images)
            images = images.reshape((len(images), sz, sz, 1)) 
            out_filename = get_dataset_path(dataset_name, imageset_id=get_imageset_id(ds))
            np.save(out_filename, images)

def read_imageset_arrays(dataset_name, sz, frac=1.0):
    """
    reads the npy files for a given dataset
    """
    slice_arrays = None
    for path, _, files in os.walk(get_dataset_path(dataset_name, sz)):
        if len(files) == 0:
            continue
        for file in files:
            if not(file.endswith('npy')):
                continue
            if (random.random() > frac):
                continue;
            dataset_fullpath = '\\'.join([path, file])
            print('loading dataset: ', dataset_fullpath)
            slice_array = np.load(dataset_fullpath)
            if (slice_arrays is None):
                slice_arrays = slice_array
            else:
                slice_arrays = np.append(slice_arrays, slice_array, axis=0)
    return slice_arrays

if __name__ == '__main__':
    # processor to create npy from dicom slices
    tcia_path = os.environ('TCIA_DATA')
    import_dataset('\\'.join([tcia_path, 'LIDC-IDRI']), 'LIDC', 60)
    import_dataset('\\'.join([tcia_path, 'AAPM-SPIE']), 'AAPM-SPIE', 60)
