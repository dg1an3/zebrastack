import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import reprlib
import logging

def grid_for_sz(sz=9):
    """ """
    return np.meshgrid(np.linspace(-(sz//2), sz//2, sz),
                       np.linspace(-(sz//2), sz//2, sz))

def make_dirac_kernel(sz=9):
    return [[0.]*sz] * (sz//2) \
            + [[0.]*(sz//2)+[1.]+[0.]*(sz//2)] \
            + [[0.]*sz] * (sz//2)

def kernel_list_to_tf(kernel_list, dtype=tf.float32):
    kernel_list = np.expand_dims(kernel_list, axis=-1)
    kernel_list = np.moveaxis(kernel_list,0,-1)
    return tf.constant(kernel_list, dtype=dtype)
    
def make_gauss_kernels(sz=9, sigmas=[0.5, 1., 2., 4.]):
    """ """
    xs, ys = grid_for_sz(sz)
    kernels = [(1 / (2 * np.pi * sigma**2)) *
                   np.exp(-(xs*xs + ys*ys) / (2.*sigma*sigma))
               for sigma in sigmas]
    return kernel_list_to_tf(kernels)

def make_dog_kernels(sz=9, sigmas=[0.5, 1., 2., 4.]):
    narrow = make_gauss_kernels(sz, sigmas)
    wide_sigmas = [sigma*2. for sigma in sigmas[:-1]] + [1e+8]
    wide = make_gauss_kernels(sz, wide_sigmas)
    dog_kernels = np.array(narrow - wide)
    return tf.constant(dog_kernels, tf.float32)

def wave_numbers(count):
    """ """
    angles_rad = [(n * np.pi/float(count)) for n in range(count)]
    return [(np.sin(angle_rad), np.cos(angle_rad)) for angle_rad in angles_rad]

def make_sine_kernels(sz=9, ks=[(1.0,0.0)], freqs=[1.0]):
    """ """    
    xs, ys = grid_for_sz(sz)
    sine_kernels = []
    for freq in freqs:
        for k in ks:        
            logging.info(f"k={k} freq={freq}")
            sine_kernels \
                .append(np.exp(freq * (xs*k[0] + ys*k[1]) * 1.0j))
    return kernel_list_to_tf(sine_kernels, dtype=tf.complex64)

def make_gabor_kernels(sz=9, ks=[(1.0,0.0)], freqs=[1.0]):
    sine_kernels = \
        make_sine_kernels(sz=sz, ks=ks, freqs=freqs)
    logging.info(f"Sine kernels shape = {sine_kernels.shape}")
    
    gauss_kernels = make_gauss_kernels(sz=sz, sigmas=[(2./f) for f in freqs])
    gauss_kernels = \
        np.repeat(gauss_kernels, 
                  sine_kernels.shape[-1] // gauss_kernels.shape[-1], axis=-1)
    logging.info(f"Gaussian kernel shape = {gauss_kernels.shape}, "
                     + f"Sine kernel shape = {sine_kernels.shape}")
    return gauss_kernels * sine_kernels

def conv2d_sq(img:tf.Tensor, filters:tf.Tensor):
    """ """
    if filters.dtype == tf.complex64:
        return tf.nn.conv2d(img, tf.math.real(filters), strides=1, padding='SAME') ** 2 \
                + tf.nn.conv2d(img, tf.math.imag(filters), strides=1, padding='SAME') ** 2
    return tf.nn.conv2d(img, filters, strides=1, padding='SAME') ** 2

def show_filter_bank(filter_bank:tf.Tensor, rows=1):
    filter_count = filter_bank.shape[-1]
    per_row = filter_count // rows
    assert rows * per_row == filter_count
    
    logging.info(f"Showing {filter_count} filters in {per_row} rows")
    fig, axs = plt.subplots(rows,per_row,figsize=(5,2*rows))
    for n in range(filter_count):
        filter = tf.reshape(filter_bank[...,0,n], filter_bank.shape[:2])
        if rows > 1:
            logging.info(f"{n}:{n // per_row} {n % per_row}")
            axs[n // per_row][n % per_row].imshow(tf.math.real(filter), cmap='plasma')
        else:
            axs[n].imshow(filter, cmap='plasma')

def show_filter_response(filter_response:tf.Tensor, rows=1):
    filter_count = filter_response.shape[-1]
    per_row = filter_count // rows
    assert rows * per_row == filter_count
    
    logging.info(f"Showing {filter_count} filters in {per_row} rows")
    fig, axs = plt.subplots(rows,per_row,figsize=(5,2*rows))
    for n in range(filter_count):
        logging.info(f"filter_response.shape = {filter_response.shape})")
        filter = tf.reshape(filter_response[...,n], 
                            (filter_response.shape[1], filter_response.shape[2]))
        if rows > 1:
            logging.info(f"{n}:{n // per_row} {n % per_row}")
            axs[n // per_row][n % per_row].imshow(filter, cmap='plasma')
        else:
            axs[n].imshow(filter, cmap='plasma')            
        