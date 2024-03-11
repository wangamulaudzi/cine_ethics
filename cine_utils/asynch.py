# Import statements
from cine_utils.api import openai_api
import streamlit as st
import cv2

# Identifying faces
from cine_utils.identify_faces import movies_to_analyse

# Image display
from cine_utils.image_display import display_characters

# Imports for morphing the characters
from cine_utils.morph import image_mixer_api, index_face
from PIL import Image

###########################
# ASYNCHRONOUS FUNCTION 1 #
###########################

def generate_synopsis_and_poster(title_1, title_2):
    ##################################
    # GENERATING SYNOPSIS AND POSTER #
    ##################################

    with st.spinner('Generating New Movie'): # Spinner tho show that it's loading
        # Call OpenAI API for synopsis and poster merging
        img, response = openai_api(title_1, title_2)

    # Displaying generated Poster
    st.title('Generated Movie Poster')
    st.image(img)

    # Displaying Generated Synopsis
    st.title('Generated Synopsis')
    merged_synopsis = response.choices[0].message.content
    st.write(merged_synopsis)


###########################
# ASYNCHRONOUS FUNCTION 2 #
###########################

def load_characters_and_merge(title_1, title_2, syn_1, syn_2):
    #################################################################
    # LOADING CHARACTER PICTURES, IMREAD OBJECTS AND BOUNDING BOXES #
    #################################################################

    with st.spinner('Loading Characters'): # Spinner tho show that it's loading
        faces_title_1, imread_faces_title_1, faces_bounds_title_1, faces_title_2, imread_faces_title_2, faces_bounds_title_2 = movies_to_analyse(title_1, title_2)

    #################################
    # DISPLAYING FIRST MOVIE IMAGES #
    #################################

    st.title(f"{title_1}")
    selected_image_1 = display_characters(faces_title_1)

    ###################################
    # DISPLAYING FIRST MOVIE SYNOPSIS #
    ###################################

    with st.expander(f"Show/hide {title_1} synopsis"): # Dropdown to hide or show sinopsis
        st.markdown(f"""{syn_1}""")


    ##################################
    # DISPLAYING SECOND MOVIE IMAGES #
    ##################################

    st.title(f"{title_2}")
    selected_image_2 = display_characters(faces_title_2)

    ####################################
    # DISPLAYING SECOND MOVIE SYNOPSIS #
    ####################################

    with st.expander(f"Show/hide {title_2} synopsis"): # Dropdown to hide or show sinopsis
        st.markdown(f"""{syn_2}""")

    # Check if both images are selected
    if selected_image_1 and selected_image_2:
        ####################
        # SELECTED IMAGE 1 #
        ####################

        # Find the index of selected_image_1 in faces_title_1
        indx_image_1 = index_face(faces_title_1, selected_image_1)

        ####################
        # SELECTED IMAGE 2 #
        ####################

        # Find the index of selected_image_2 in faces_title_2
        indx_image_2 = index_face(faces_title_1, selected_image_1)

        # Save the selected images
        path_1 = "raw_data/morph/selected_image_1.png"
        cv2.imwrite(path_1, imread_faces_title_1[indx_image_1])

        path_2 = "raw_data/morph/selected_image_2.png"
        cv2.imwrite(path_2, imread_faces_title_2[indx_image_2])

        ####################
        # MORPH THE IMAGES #
        ####################

        with st.spinner('Generating New Character'): # Spinner tho show that it's loading

            morph_path = image_mixer_api(path_1, path_2)

            # Open the image using PIL
            morphed_image = Image.open(morph_path)

            st.title('Generated Character')

            # Display the image in Streamlit
            st.image(morphed_image)
