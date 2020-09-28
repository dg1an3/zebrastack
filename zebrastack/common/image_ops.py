import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import img_as_uint
from cv2 import createCLAHE

def img2grayscale(img):
    """ convert an array-like image to grayscale """
    if len(img.shape) > 2:
        return rgb2gray(img)
    return img

def whiten_img(img):
    """ whiten an image, so it is between 0.0 and 1.0 """
    width = np.max(img) - np.min(img)
    img = img - np.min(img)
    img = img/width
    return img

def resize_img(img, sz=128):
    """ resize an image, include turning it into a uint array """
    img = resize(img, (sz,sz))
    img = whiten_img(img)
    img = img_as_uint(img)
    return img

clahe = createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def clahe_img(img):
    """ clahe filter an array-like image """
    img = clahe.apply(img)
    return img

def center_surround(original_img, sz=128):
    """ full center-surround
    * converting to grayscale
    * resize to common size,
    * applying CLAHE
    * whiten
    """
    img = img2grayscale(original_img)
    img = resize_img(img, sz)
    img = clahe_img(img)
    img = whiten_img(img)
    return img

def channelized_center_surround(original_img, filter_bank, sz=128):
    """ center-surround processing channelized by a filter bank """
    img = center_surround(original_img)
    img = np.reshape(img, (1,sz,sz,1))    
    img_tf = tf.constant(img, tf.float32)
    img_out = tf.nn.conv2d(img_tf, filter_bank,
                           strides=[1, 1, 1, 1],
                           padding='SAME')
    return img_out