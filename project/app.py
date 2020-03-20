from flask import Flask, render_template, send_from_directory, request, send_file
import json
import io
import numpy as np
from skimage.io import imsave

app = Flask(__name__)

@app.route('/hello')
def hello():
    return render_template('hello.html')

@app.route('/slider')
def slider():
    return render_template('slider.html')

@app.route('/sliderValue')
def sliderValue():
    values = request.args['newValue']
    text = '|'.join([('*' * (int(n) // 10)) for n in values.split(',')])
    print(text)
    return text

@app.route('/imageUpdate')
def imageUpdate():
    """
    Return a generated image as a png by
    saving it into a StringIO and using send_file.
    """
    values = request.args['newValue']
    print(values)

    num_tiles = 20
    tile_size = 30
    arr = np.random.randint(0, 255, (num_tiles, num_tiles, 3))
    arr = arr.repeat(tile_size, axis=0).repeat(tile_size, axis=1)

    # We make sure to use the PIL plugin here because not all skimage.io plugins
    # support writing to a file object.
    bytesIO = io.BytesIO()
    imsave(bytesIO, arr, plugin='pil', format_str='png')
    bytesIO.seek(0)
    return send_file(bytesIO, mimetype='image/png')

    # filename = 'test.png'
    # return send_file(filename, mimetype='image/png')    

@app.route('/jsx/<path:path>')
def send_jsx(path):
    return send_from_directory('jsx', path)

if __name__ == '__main__':
    app.run(debug=True)