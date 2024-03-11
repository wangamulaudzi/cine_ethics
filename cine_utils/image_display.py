from PIL import Image
import streamlit as st

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

    return char_list
