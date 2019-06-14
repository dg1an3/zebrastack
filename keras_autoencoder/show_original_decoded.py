import matplotlib.pyplot as plt

def show_grayscale(rows, columns, at, pixel_array, sz):
    ax = plt.subplot(rows, columns, at)
    plt.imshow(pixel_array.reshape(sz, sz))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def show_original_decoded(original, decoded, sz):
    n = 10  # how many digits we will display
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        show_grayscale(2, n, i+1, original[i], sz)
        show_grayscale(2, n, i+1+n, decoded[i], sz)
    plt.show(block=True)

