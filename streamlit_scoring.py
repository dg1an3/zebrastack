import streamlit as st
import os
import random
import yaml
from datetime import datetime
from PIL import Image

# Set page config for wider layout
st.set_page_config(page_title="Image Selection App", layout="wide")


# Function to randomly select a folder and three images
def select_images(base_path):
    folders = [f.path for f in os.scandir(base_path) if f.is_dir()]
    if not folders:
        st.error("No folders found in the specified directory.")
        return None, None

    selected_folder = random.choice(folders)
    images = [
        f
        for f in os.listdir(selected_folder)
        if f.lower().endswith(("png", "jpg", "jpeg"))
    ]
    if len(images) < 3:        
        # st.error(f"Not enough images in folder: {selected_folder}")
        # return None, None
        # Instead of erroring out, just try another folder
        return select_images(base_path)

    selected_images = random.sample(images, 3)
    return selected_folder, selected_images


# Function to save the selection to a YAML file
def save_selection(folder, images, selected_image):
    timestamp = datetime.now().isoformat()
    data = {
        "timestamp": timestamp,
        "folder": folder,
        "images": images,
        "selected_image": selected_image,
    }
    # Create unique filename with timestamp to avoid overwriting
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
    yaml_file = os.path.join(folder, f"selection_{timestamp_str}.yaml")
    with open(yaml_file, "w") as file:
        yaml.dump(data, file)


# Streamlit app
st.title("Image Selection App")

# Add slider for image width control
image_width = st.slider("Image Width", min_value=150, max_value=600, value=300, step=25)

# st.text_input("Enter the base directory path:", value=".")
base_path = "screen_captures" 

if True: # st.button("Select Images"):
    folder, images = select_images(base_path)
    if folder and images:
        st.session_state["folder"] = folder
        st.session_state["images"] = images
        st.session_state["selected_image"] = None

if "images" in st.session_state:
    folder = st.session_state["folder"]
    images = st.session_state["images"]

    cols = st.columns(3)
    for i, col in enumerate(cols):
        image_path = os.path.join(folder, images[i])
        with col:
            st.image(Image.open(image_path), caption=f"Image {i+1}", width=image_width)
            if st.button(f"Select Image {i+1}"):
                st.session_state["selected_image"] = images[i]
                save_selection(folder, images, images[i])
                st.success(f"You selected Image {i+1}. Selection saved!")
                
                # Automatically generate new random path and select three new images
                new_folder, new_images = select_images(base_path)
                if new_folder and new_images:
                    st.session_state["folder"] = new_folder
                    st.session_state["images"] = new_images
                    st.session_state["selected_image"] = None
                    st.rerun()  # Refresh the page to show new images
