# Import statements
import streamlit as st
import pandas as pd

# API for the search bar
from cine_utils.api import endpoint
from cine_utils.api import openai_api

# Identifying faces
from cine_utils.identify_faces import movies_to_analyse

# Image display
from cine_utils.image_display import display_characters

# Imports for loading the big dataframe
from dotenv import load_dotenv
import os

# Imports for morphing the characters
from cine_utils.morph import index_face, image_mixer_api
import cv2
from PIL import Image

#loading credentials
load_dotenv()

###################
# LOADING DATASET #
###################

file_path = os.getenv("CINE_PICK_TABLE")
df = pd.read_csv(file_path, sep=',')


                        ####################
                        # STREAMLIT LAYOUT #
                        ####################

################
# SIDEBAR MENU #
################

#st.title('Movie Synopsis Merger')
st.sidebar.image('raw_data/logo/CinePickSmall.png')#, caption='Cine Pick')


############################
# DEFINING JAVASCRIPT CODE #
############################

# Javascript as python string
javascript_code = """
<script>
var input = document.getElementById('textfield');
var datalist = document.getElementById('suggestions');

input.addEventListener('input', function(e) {
    var query = input.value;
    fetch('/suggest?query=' + query)
        .then(response => response.json())
        .then(data => {
            datalist.innerHTML = '';
            data.forEach(function(suggestion) {
                var option = document.createElement('option');
                option.value = suggestion;
                datalist.appendChild(option);
            });
        });
});
</script>
"""

# Render the JavaScript code using st.markdown
st.markdown(javascript_code, unsafe_allow_html=True)

page_bg_img = '''<style>body {background-image: "raw_data/logo/CinePickSmall.png"; background-size: cover;}</style>'''

st.markdown(page_bg_img, unsafe_allow_html=True)


# Placeholder for displaying suggestions
st.write("<datalist id='suggestions'></datalist>", unsafe_allow_html=True)

# Create an endpoint for the suggestion's search bar
endpoint(df)



###################################
# MOVIE SELECTION SIDEBAR SECTION #
###################################

# Streamlit layout for sidebar genre selection functionality
st.sidebar.title('Movie Synopsis Merger 🍿🎬')  # Title
st.sidebar.caption("Discover your own innovative plot!")  # Description


# Select Genre Radio Button
st.markdown("""<style>span[data-baseweb="tag"] {  background-color: red !important;}</style>""", unsafe_allow_html=True)
genre = st.sidebar.multiselect("Choose the movie genre", df['genre'].unique(), default=df['genre'].unique())

data = df.loc[df['genre'].isin(genre)]


# Multiselect field for selecting movies
selected_indices = st.sidebar.multiselect('Select two movies to merge:',
                                        data.index,
                                        format_func=lambda x: data['title'].loc[x].title())


# Check if more than two movies are selected
if len(selected_indices) > 2:
    st.sidebar.warning('Please select at most two movies.')
    # Limit the selected indices to the first two selected indices
    selected_indices = selected_indices[:2]




#######################
# RUNNING APPLICATION #
#######################

# When 'Merge Synopsis' button is pushed
if st.sidebar.button('Merge Movies') and len(selected_indices) == 2:
    title_1, title_2 = df['title'].loc[selected_indices].values
    title_1, title_2 = title_1.title(), title_2.title()

    # Getting original synopsis of the selected movies.
    syn_1, syn_2 = df['summarized_synopsis'].loc[selected_indices].values


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
