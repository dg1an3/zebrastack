import matplotlib.pyplot as plt
from scipy import ndimage
from IPython.display import clear_output
from image_ops import img2grayscale, resize_img

def show_thumbnail(rows, columns, at, pixel_array, sz=128):
    """ shows a single thumbnail at a given positioin in the subplot """
    ax = plt.subplot(rows, columns, at)
    img = img2grayscale(pixel_array)
    img = resize_img(img, sz)
    
    img = ndimage.zoom(img, 4.0, order=5)
    plt.imshow(img.reshape(sz*4, sz*4), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
def show_all_thumbnails(upper, lower, sz=128, n=10):
    """ show all thumbnails in two lists, one above the other """
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        show_thumbnail(2, n, i+1, upper[i], sz)
        show_thumbnail(2, n, i+1+n, lower[i], sz)
    plt.show(block=True)
    
def show_inline_notebook(upper, lower):
    """ show thumbnails inline """
    clear_output(wait=True)
    show_all_thumbnails(upper, lower, 128)
    