import os, time
import pydicom
from pydicom.data import get_testdata_files
import matplotlib
import matplotlib.pyplot as plt

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
    dataset_directory = "E:\\Data\\tcia\\LIDC-IDRI"
    datasets = [(path, files) for path, _, files in os.walk(dataset_directory) if len(files) > 0]
    for path, files in datasets:
        dss = [pydicom.dcmread(path+ "\\"+ file) for file in files]
        dss.sort(key=lambda ds:float(ds.ImagePositionPatient[2]))
        for ds in dss:
            print(ds.filename, ds.ImagePositionPatient)
            plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 
            plt.pause(0.1)

print("done")
