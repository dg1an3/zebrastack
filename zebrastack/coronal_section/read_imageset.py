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

def read_slice(full_path, sz):
    """
    reads a DICOM slice given the file full path
    """
    if not(full_path.endswith('dcm')):
        return None, None, None
    ds = pydicom.dcmread(full_path)
    if not(hasattr(ds, 'ImagePositionPatient')):
        return None, None, None

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

    slice_position = float(ds.ImagePositionPatient[2])
    return ds, slice_array, slice_position

def import_dataset(path_name, dataset_name, sz):
    """
    import_dataset('e:\\Data\\tcia\\LIDC-IDRI', 'LIDC', 60)
    """
    for path, _, files in os.walk(path_name):
        if len(files) == 0:
            continue
        images, ds = {}, None
        for file in files:
            next_ds, slice_array, slice_position = read_slice('\\'.join([path, file]), sz)
            if not(slice_array is None):
                ds = next_ds                
                images[slice_position] = slice_array
        if (len(images) > 0):
            sorted_images = [images[key] for key in sorted(images)]
            np_images = np.array(sorted_images)
            np_images = np_images.reshape((len(sorted_images), sz, sz, 1)) 
            out_filename = get_dataset_path(dataset_name, sz, npy_fname=get_imageset_npy_fname(ds))
            np.save(out_filename, np_images)

def gen_imageset_array(dataset_name, sz, frac=1.0):
    """
    reads the npy files for a given dataset
    """
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
            nparr = np.load(dataset_fullpath)
            mid_slice = int(nparr.shape[0]/2)
            # nparr = nparr[mid_slice-10:mid_slice+10]
            nparr = nparr[mid_slice:mid_slice+1]
            yield nparr

def read_imageset_arrays(dataset_name, sz, frac=1.0):
    """
    reads the npy files for a given dataset
    """
    slice_arrays = gen_imageset_array(dataset_name, sz, frac)
    slice_arrays = list(slice_arrays)
    super_array = np.concatenate(slice_arrays)
    return super_array

if __name__ == '__main__':
    # processor to create npy from dicom slices
    tcia_path = os.environ['TCIA_DATA']
    # import_dataset('\\'.join([tcia_path, 'LIDC-IDRI']), 'LIDC-IDRI', 128)
    # import_dataset('\\'.join([tcia_path, 'SPIE-AAPM Lung CT Challenge']), 'SPIE-AAPM', 128)
    import_dataset('\\'.join([tcia_path, 'SPIE-AAPM Lung CT Challenge']), 'SPIE-AAPM', 256)

