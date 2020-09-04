import pprint
import numpy as np
import math
from scipy.optimize import minimize
from tensorflow.keras.preprocessing.image import apply_affine_transform as transform

use_rotation, use_shear = False, False
dump = False
weight_prior = 1.0/1000.0
count_imgs = 0

def xform_params(x):
    params = {'tx':x[0], 'ty':x[1], 'zx':math.exp(x[2]/30), 'zy':math.exp(x[3]/30)}
    if use_rotation:
        params['theta'] = x[4]
    if use_shear:
        params['shear'] = x[5]
    return params

def xform_image(params, img):
    return transform(img, **params, fill_mode='nearest')

def xform_prior_logsq(x):
    theta_prior = x[0] * x[0]
    shear_prior = x[5] * x[5]
    scale_prior = x[3] * x[3] + x[4] * x[4]
    return theta_prior + shear_prior + scale_prior

def xformed_latent(x, img, encoder):
    params = xform_params(x)
    xformed = xform_image(params, img)
    latent_vector = encoder.predict([[xformed]])[2]
    # prior = xform_prior_logsq(x)
    return np.linalg.norm(latent_vector) # + weight_prior * prior
    
def shift(img, encoder):
    global count_imgs
    res = minimize(xformed_latent, np.zeros((4,)),
                   method='powell', # method='nelder-mead', 
                   args=(img, encoder),
                   options={'xtol': 1e-6, 
                            'ftol': 1e-6,
                            # 'maxiter': 10000,
                            'disp': dump})
    params = xform_params(res.x)
    if dump:
        print(res)
        pprint.pprint(params)
    count_imgs = 1 + count_imgs
    print('{0}: {1},{2}'.format(count_imgs, res.x[0], res.x[1]), end = '\r')
    return xform_image(params, img)
