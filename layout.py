# Import statements
import streamlit as st
import pandas as pd

# API for the search bar
from cine_utils.api import endpoint
from cine_utils.api import openai_api

# Identifying faces
from cine_utils.identify_faces import movies_to_analyse

# Image display
from streamlit_image_select import image_select
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

# Placeholder for displaying suggestions
st.write("<datalist id='suggestions'></datalist>", unsafe_allow_html=True)

# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://trello.com/1/cards/65e743e6caf555f37ac39d9b/attachments/65eeeff70aef81df2911f2fe/download/CinePickSmall_15.png");
    background-size: 75vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)


# Create an endpoint for the suggestion's search bar
endpoint(df)

################################
# INITIALIZING STATE VARIABLES #
################################

if 'movies' not in st.session_state:
    st.session_state.movies = []
    st.session_state._movies = st.session_state.movies

if 'gen_movie' not in st.session_state:
    st.session_state.gen_movie = ''
    st.session_state.gen_synopsis = ''

if 'faces_title_1' not in st.session_state:
    st.session_state.faces_title_1 = []
    st.session_state.imread_faces_title_1 = []
    st.session_state.faces_bounds_title_1 = []

if 'faces_title_2' not in st.session_state:
    st.session_state.faces_title_2 = []
    st.session_state.imread_faces_title_2 = []
    st.session_state.faces_bounds_title_2 = []

if 'title_1_char' not in st.session_state:
    st.session_state.title_1_char = []

if 'title_2_char' not in st.session_state:
    st.session_state.title_2_char = []

if 'char_1' not in st.session_state:
    st.session_state.char_1 = ''

if 'char_2' not in st.session_state:
    st.session_state.char_2 = ''

st.session_state.movies = st.session_state._movies


def key_protect():
    st.session_state._movies = st.session_state.movies


def get_genres():
    genres = []
    if crime:
        genres.append('crime')
    if thriller:
        genres.append('thriller')
    if fantasy:
        genres.append('fantasy')
    if scifi:
        genres.append('scifi')
    if romance:
        genres.append('romance')
    if family:
        genres.append('family')
    if action:
        genres.append('action')
    if adventure:
        genres.append('adventure')
    if horror:
        genres.append('horror')
    if mistery:
        genres.append('mistery')
    return genres


###################################
# MOVIE SELECTION SIDEBAR SECTION #
###################################

# Sidebar Title
st.sidebar.title('Movie Synopsis Merger 🍿🎬')  # Title
st.sidebar.caption("Discover your own innovative plot!")  # Description


<<<<<<< HEAD
# Select Genre Radio Button
st.markdown("""<style>span[data-baseweb="tag"] {  background-color: red !important;}</style>""", unsafe_allow_html=True)
genre = st.sidebar.multiselect("Choose the movie genre", df['genre'].unique(), default=df['genre'].unique())
=======
# Genre Selection splitted in 2 columns
st.sidebar.write('Select Genres:')
col1, col2 = st.sidebar.columns(2)
>>>>>>> 821240b7b237df7d360ee1e85f09343226117b5e

with col1:
    crime = st.checkbox('crime', value=False)
    thriller = st.checkbox('thriller', value=False)
    fantasy = st.checkbox('fantasy', value=False)
    scifi = st.checkbox('scifi', value=False)
    romance = st.checkbox('romance', value=False)

with col2:
    family = st.checkbox('family', value=False)
    action = st.checkbox('action', value=False)
    adventure = st.checkbox('adventure', value=False)
    horror = st.checkbox('horror', value=False)
    mistery = st.checkbox('mistery', value=False)


# Getting selected Genres
genres = get_genres()

# Getting Database with only selected genres
data = df.loc[df['genre'].isin(genres)]

try:
    # Multiselect field for selecting movies
    st.sidebar.write('')
    st.sidebar.write('Select Movies:')
    st.sidebar.multiselect('movies', data.index,
                        format_func=lambda x: data['title'].loc[x].title(),
                        key='movies',
                        on_change=key_protect,
                        label_visibility="collapsed")
except Exception as e:
    st.sidebar.warning('Cannot remove Genre that already has a selected movie.')


# getting session variable
selected_indices = st.session_state.movies


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
    if st.session_state.gen_movie == '' or st.session_state.gen_synopsis == '':
        with st.spinner('Generating New Movie'): # Spinner tho show that it's loading
            # Call OpenAI API for synopsis and poster merging
            img, response = openai_api(title_1, title_2)
            st.session_state.gen_movie = img
            st.session_state.gen_synopsis = response.choices[0].message.content


    # Displaying generated Poster
    st.title('Generated Movie Poster')
    st.image(st.session_state.gen_movie)

    # Displaying Generated Synopsis
    st.title('Generated Synopsis')
    merged_synopsis = st.session_state.gen_synopsis
    st.write(merged_synopsis)



    #################################################################
    # LOADING CHARACTER PICTURES, IMREAD OBJECTS AND BOUNDING BOXES #
    #################################################################

    if st.session_state.faces_title_1 == [] or st.session_state.faces_title_2 == []:
        with st.spinner('Loading Characters'): # Spinner to show that it's loading
            faces_title_1, imread_faces_title_1, faces_bounds_title_1, faces_title_2, imread_faces_title_2, faces_bounds_title_2 = movies_to_analyse(title_1, title_2)
            st.session_state.faces_title_1 = faces_title_1
            st.session_state.imread_faces_title_1 = imread_faces_title_1
            st.session_state.faces_bounds_title_1 = faces_bounds_title_1
            st.session_state.faces_title_2 = faces_title_2
            st.session_state.imread_faces_title_2 = imread_faces_title_2
            st.session_state.faces_bounds_title_2 = faces_bounds_title_2

    #################################
    # DISPLAYING FIRST MOVIE IMAGES #
    #################################

    st.title(f"{title_1}")
    st.session_state.title_1_char = display_characters(st.session_state.faces_title_1)

    # Display images as clickable boxes
    img_1 = image_select(label='char_1',
                        images=st.session_state.title_1_char,
                        center = False,
                        width = 455,
                        height = 256,
                        use_container_width=True,
                        return_value='original',
                        label_visibility = 'hidden')

    st.session_state.char_1 = img_1

    ###################################
    # DISPLAYING FIRST MOVIE SYNOPSIS #
    ###################################

    with st.expander(f"Show/hide {title_1} synopsis"): # Dropdown to hide or show sinopsis
        st.markdown(f"""{syn_1}""")


    ##################################
    # DISPLAYING SECOND MOVIE IMAGES #
    ##################################

    st.title(f"{title_2}")
    st.session_state.title_2_char = display_characters(st.session_state.faces_title_2)

    # Display images as clickable boxes
    img_2 = image_select(label='char_2',
                                    images=st.session_state.title_2_char,
                                    center = False,
                                    width = 455,
                                    height = 256,
                                    use_container_width=True,
                                    return_value='original',
                                    label_visibility = 'hidden')

    st.session_state.char_2 = img_2
    ####################################
    # DISPLAYING SECOND MOVIE SYNOPSIS #
    ####################################

    with st.expander(f"Show/hide {title_2} synopsis"): # Dropdown to hide or show sinopsis
        st.markdown(f"""{syn_2}""")


    # Check if both images are selected
    #if selected_image_1 and selected_image_2:
    # if st.button('Merge Characters'):

    ####################
    # SELECTED IMAGE 1 #
    ####################

    # Find the index of selected_image_1 in faces_title_1
    indx_image_1 = index_face(st.session_state.title_1_char, st.session_state.char_1)

    ####################
    # SELECTED IMAGE 2 #
    ####################

    # Find the index of selected_image_2 in faces_title_2
    indx_image_2 = index_face(st.session_state.title_2_char, st.session_state.char_2)

    # Save the selected images
    path_1 = "raw_data/morph/selected_image_1.png"
    cv2.imwrite(path_1, st.session_state.imread_faces_title_1[indx_image_1])

    path_2 = "raw_data/morph/selected_image_2.png"
    cv2.imwrite(path_2, st.session_state.imread_faces_title_2[indx_image_2])

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
