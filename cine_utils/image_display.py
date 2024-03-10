from PIL import Image
import streamlit as st
from streamlit_image_select import image_select

def display_characters(faces_title):
    # Create a list of images
    char_list = []
    for image_char in faces_title:
        # Extract the NumPy array representing the image data
        image_data = image_char[0]

        # Convert the NumPy array to a PIL image object
        pil_image = Image.fromarray(image_data, 'RGB')
        char_list.append(pil_image)
        #char_list.append(image_data)

    # Display images as clickable boxes
    selected_images = image_select(label='Select Image to Merge',
                                    images=char_list,
                                    captions=None,
                                    index=0,
                                    center = False,
                                    width = 455,
                                    height = 256,
                                    use_container_width=True,
                                    return_value='original',
                                    key=None,
                                    label_visibility = 'hidden')
    return selected_images
