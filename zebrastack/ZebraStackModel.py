import sys
from pathlib import Path
from typing import Union, Optional, Tuple
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def sampling_layer(args:Tuple[tf.Tensor,tf.Tensor]) -> tf.Tensor:
    """reparameterization trick by sampling fr an isotropic unit Gaussian.
    instead of sampling from Q(z|X), sample eps = N(0,I) 
        then z = z_mean + sqrt(var)*eps

    Args:
        args (tuple[tf.Tensor,tf.Tensor]): [description]

    Returns:    
        tf.Tensor: sampled latent vector        
    """
    z_mean, z_log_var = args
    
    # from keras import backend as K
    batch, dim = K.shape(z_mean)[0], K.int_shape(z_mean)[1]

    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon    

def vae_loss(z_mean, z_log_var, y_true, y_pred, use_mse=True):
    """compute VAE loss, using either mse or crossentropy.

    Args:
        z_mean: mean of Q(z|X)
        z_log_var: log variance of Q(z|X)
        y_true, y_pred: truth and predicated values

    Returns:
        loss value
    """
    from tensorflow.keras.losses import mse, binary_crossentropy
    # from keras import backend as K
    sz = 128
    img_pixels = sz * sz # * 100.0
    match_loss = img_pixels * \
        mse(K.flatten(y_true), K.flatten(y_pred)) if use_mse \
        else binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))

    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return match_loss + (1e-2 * kl_loss)

class ZebraStackModel(object):
    """class the wraps a zebrastack keras model
    """

    def __init__(self, sz=128, latent_dim=8):
        """initialize the zebrastack model

        Args:
            sz (int, optional): [description]. Defaults to 128.
            latent_dim (int, optional): [description]. Defaults to 8.
        """

        self.latent_dim = latent_dim
        self.sz = sz

        with open("zebrastack_encoder_1.yaml") as encoder_file:
            self.encoder = tf.keras.models.model_from_yaml(encoder_file)

        with open("zebrastack_decoder_1.yaml") as decoder_file:
            self.decoder = tf.keras.models.model_from_yaml(decoder_file)

        self.autoencoder = None

    def train(self, dataset:Union[str,Path,tf.data.Dataset,None], apply_shifter:Optional[int]=None):
        """[summary]

        Args:
            dataset (Union[str,Path,Dataset,None]): [description]
            apply_shifter (Optional[int], optional): [description]. Defaults to None.
        """

        # get training dataset

        # update during training -- callback or tensorboard?

        # save checkpoints during training

        # now save after the training is complete

        # save onnx of decoder after the training is complete

        return True

    def recognize(self, img:np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            img (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """

        return np.zeros(self.latent_dim)

    def generate(self, latent_vec:np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            latent_vec (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """

        # TODO: also provide final size and diffusion parameters for reconstruction

        return np.zeros([self.sz, self.sz])

if __name__=='__main__':
    import sys
    opts = [opt for opt in sys.argv[1:]]

    if "-t" in opts:
        # perform training on dataset provided
        print("training")
    elif "-r" in opts:
        # perform recognition on an image file
        print("recognition")
    elif "-g" in opts:
        # perform generation on vector passed
        print("generating")
    else:
        raise SystemExit(f"Usage: {sys.argv[0]} (-t | -r | -g) <arguments>...")
