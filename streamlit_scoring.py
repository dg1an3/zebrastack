import streamlit as st
import os
import random
import yaml
from datetime import datetime
from PIL import Image


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
        st.error(f"Not enough images in folder: {selected_folder}")
        return None, None

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
    yaml_file = os.path.join(folder, "selection.yaml")
    with open(yaml_file, "w") as file:
        yaml.dump(data, file)


# Streamlit app
st.title("Image Selection App")

base_path = st.text_input("Enter the base directory path:", value=".")

if st.button("Select Images"):
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
            st.image(Image.open(image_path), caption=f"Image {i+1}")
            if st.button(f"Select Image {i+1}"):
                st.session_state["selected_image"] = images[i]
                save_selection(folder, images, images[i])
                st.success(f"You selected Image {i+1}. Selection saved!")
