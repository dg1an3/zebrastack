import os
from pathlib import Path, PurePath
import numpy as np
from typing import Callable, List
from imageio import imread
from pydicom import dcmread
from .show_thumbnails import show_inline_notebook

# set up the data_all and data_temp variables
data_all, data_temp = Path(os.environ['DATA_ALL']), Path(os.environ['DATA_TEMP'])
print(f"image_processing: DATA_ALL={data_all}; DATA_TEMP={data_temp}")

def temp_from_original(original_path:Path, temp_label):
    """
    forms the corresponding temp path from an original path
    """    
    temp_path = data_temp / original_path.relative_to(data_all).parts[0] / temp_label
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path

def preprocess_images(filepaths:List[str], reader:Callable, 
                      processor:Callable, temp_path:Path, 
                      show_in_notebook=True):
    """ performs preprocessing on a list of filepaths
    storing in numpy files in the temp path
    """
    if show_in_notebook:
        original_imgs, processed_imgs = [], []
    for filepath in filepaths:
        relative_path, original_img = reader(filepath)        
        processed_path = temp_path / relative_path
        if processed_path.with_suffix('.npy').exists():
            print(f"Skipping path {str(processed_path)}", end='\r')
            continue
        
        # ensure the processed path exists
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        # perform processing
        processed_img = processor(original_img)
        
        # increment original and processed arrays
        if show_in_notebook:
            original_imgs.append(original_img)        
            processed_imgs.append(processed_img)
        
        # save processed to the temp path
        np.save(processed_path, processed_img)
        
        # output a message, and show callback if it is time
        print(f"{processed_path} {original_img.shape} {original_img.dtype}", end='\r')
        if (len(processed_imgs) == 100):
            if show_in_notebook:
                show_inline_notebook(original_imgs[-11:-1], processed_imgs[-11:-1])
            original_imgs, processed_imgs = [], []
                       
def read_imageio(filepath:str):
    """ helper to read a standard img (jpg, png, etc) 
    and return a suitable temp relative path """
    original_img = imread(filepath)
    relative_path = PurePath(Path(filepath).stem)
    return relative_path, original_img

def read_dcm(filepath:str):
    """ helper to read a DICOM image 
    and return a suitable temp relative path """
    dcm = dcmread(str(filepath), defer_size=128)
    relative_path = \
        PurePath(dcm.PatientID) / \
            f"{dcm.Modality}-{dcm.SeriesInstanceUID}" / \
            f"{dcm.InstanceNumber:05d}"
    return relative_path, dcm.pixel_array