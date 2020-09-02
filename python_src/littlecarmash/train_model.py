#!/usr/bin/env python
# coding: utf-8

# In[1]:


from processed_image import ProcessedImage, read_from_dir, show_image_strip
imgs = list(read_from_dir('..\\Data\\LittleCarDb1'))
print([str(img) for img in imgs[0:5]])


# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
_, axes = plt.subplots(1, 10, sharey=True, figsize=(8,1.5))
show_image_strip(imgs, axes)


# In[3]:


_, axes = plt.subplots(2, 10, sharey=True, figsize=(8,1.5))
processed_dict = {img.fullpath:img.get_processed_image() for img in imgs}
show_image_strip(imgs, axes, predicted_dict=processed_dict)


# In[15]:


import numpy as np
import random
random.shuffle(imgs)
x_train = [img.get_processed_image(size=128) for img in imgs]
x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))

test_size = int(len(x_train)/10)
x_test = x_train[0:test_size]
x_train = x_train[test_size+1:]
print(x_train.shape, x_test.shape)


# In[16]:


from model_vae_3stage import ModelVae3Stage
all_model = ModelVae3Stage(size=128, in_channels=1, latent_dim=8)
vae, enc, dec = all_model.vae, all_model.encoder, all_model.decoder
vae.summary()


# In[17]:


vae.fit(x_train, x_train, epochs=10, batch_size=1024, 
        shuffle=True, validation_data=(x_test,x_test))


# In[18]:


def show_original_decoded(original, decoded, sz):
    n = 10  # how many digits we will display
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(original[i].reshape(128, 128), cmap='gray')
        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(decoded[i].reshape(128, 128), cmap='gray')
    plt.show()


# In[19]:


encoded_latent = enc.predict(x_test)[2]   # z parameter is #2
decoded_imgs = dec.predict(encoded_latent)
print(x_test.shape, '->', encoded_latent.shape, '->', decoded_imgs.shape)
show_original_decoded(x_test, decoded_imgs, 128)


# In[45]:


vae.fit(x_train, x_train, epochs=400, batch_size=1024, 
        shuffle=True, validation_data=(x_test,x_test))


# In[46]:


encoded_latent = enc.predict(x_test)[2]   # z parameter is #2
decoded_imgs = dec.predict(encoded_latent)
print(x_test.shape, '->', encoded_latent.shape, '->', decoded_imgs.shape)
show_original_decoded(x_test, decoded_imgs, 128)


# In[33]:


vae.fit(x_train, x_train, epochs=40, batch_size=1024, 
        shuffle=True, validation_data=(x_test,x_test))


# In[34]:


encoded_latent = enc.predict(x_test)[2]   # z parameter is #2
decoded_imgs = dec.predict(encoded_latent)
print(x_test.shape, '->', encoded_latent.shape, '->', decoded_imgs.shape)
show_original_decoded(x_test, decoded_imgs, 128)


# In[35]:


vae.fit(x_train, x_train, epochs=100, batch_size=1024, 
        shuffle=True, validation_data=(x_test,x_test))


# In[25]:


encoded_latent = enc.predict(x_test)[2]   # z parameter is #2
decoded_imgs = dec.predict(encoded_latent)
print(x_test.shape, '->', encoded_latent.shape, '->', decoded_imgs.shape)
show_original_decoded(x_test, decoded_imgs, 128)


# In[26]:


vae.fit(x_train, x_train, epochs=40, batch_size=1024,
        shuffle=True, validation_data=(x_test,x_test))


# In[36]:


encoded_latent = enc.predict(x_test)[2]   # z parameter is #2
decoded_imgs = dec.predict(encoded_latent)
print(x_test.shape, '->', encoded_latent.shape, '->', decoded_imgs.shape)
show_original_decoded(x_test, decoded_imgs, 128)


# In[37]:


vae.fit(x_train, x_train, epochs=100, batch_size=1024, 
        shuffle=True, validation_data=(x_test,x_test))


# In[38]:


encoded_latent = enc.predict(x_test)[2]   # z parameter is #2
decoded_imgs = dec.predict(encoded_latent)
print(x_test.shape, '->', encoded_latent.shape, '->', decoded_imgs.shape)
show_original_decoded(x_test, decoded_imgs, 128)


# In[ ]:


vae.fit(x_train, x_train, epochs=100, batch_size=1024, 
        shuffle=True, validation_data=(x_test,x_test))
vae.save('model_vae_3stage.h5')


# In[ ]:


encoded_latent = enc.predict(x_test)[2]   # z parameter is #2
decoded_imgs = dec.predict(encoded_latent)
print(x_test.shape, '->', encoded_latent.shape, '->', decoded_imgs.shape)
show_original_decoded(x_test, decoded_imgs, 128)


# In[ ]:


vae.fit(x_train, x_train, epochs=1000, batch_size=1024, 
        shuffle=True, validation_data=(x_test,x_test))
vae.save('model_vae_3stage.h5')


# In[ ]:


encoded_latent = enc.predict(x_test)[2]   # z parameter is #2
decoded_imgs = dec.predict(encoded_latent)
print(x_test.shape, '->', encoded_latent.shape, '->', decoded_imgs.shape)
show_original_decoded(x_test, decoded_imgs, 128)


# In[ ]:


with open("model_vae_3stage.yaml", "w") as yaml_model_file:
    yaml_model_file.write(vae.to_yaml())

