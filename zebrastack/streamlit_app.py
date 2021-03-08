import streamlit as st
import numpy as np
import pandas as pd

"""
# My first app
Here's our first attempt at using data to create a table:
"""

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

# chart_data = pd.DataFrame(
#      np.random.randn(20, 3),
#      columns=['a', 'b', 'c'])

# st.line_chart(chart_data)

# map_data = pd.DataFrame(
#     np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#     columns=['lat', 'lon'])

# st.map(map_data)

option = st.sidebar.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected:', option

pixel_b = st.sidebar.slider('brightness', 0, 200, 100)  # min: 0h, max: 23h, default: 17h

@st.cache()
def gen_image(for_pixel_b):

    image = np.array([[1,2,3,0,0],
                    [4,50,for_pixel_b,7,8],
                    [9,10,110,for_pixel_b,13],
                    [15,16,170,18,19]])
    return image


st.image(gen_image(pixel_b), caption="some image")