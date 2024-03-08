# Import statements
import streamlit as st
import pandas as pd

# Search bar import statements
import uvicorn

# API for the search bar
from api import endpoint
from api import openai_api

# Identifying faces
from identify_faces import movies_to_analyse
from PIL import Image


# Load New summarization stuff
from transformers import pipeline
# Initialize the summarization pipeline with the BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# Functions
# Function to filter DataFrame based on search query
def filter_options(search_query, max_suggestions=5):
    """
    Creates a suggestions dataframe based on the user's search_query

    Returns:
        DataFrame
    """
    suggestions = df[df['title'].str.contains(search_query, case=False)][:max_suggestions]
    return suggestions['title'].tolist()

# Streamlit functionality
# Load dataset
file_path = 'raw_data/synopsis_screengrabs_final_table.csv'  # Update to your file path
df = pd.read_csv(file_path, sep=',')

# Streamlit layout
#st.title('Movie Synopsis Merger')
st.sidebar.image('raw_data/logo/CinePickSmall.png')#, caption='Cine Pick')

# Sidebar for search functionality
# #st.sidebar.title("Search & Select Movies")
# search_query = st.sidebar.text_input('Search movie titles:', key="search_input")

# Define the JavaScript code as a string
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

# Create an endpoint for the suggestion's search bar
endpoint(df)

# Multiselect for selecting movies
selected_indices = st.sidebar.multiselect('Select two movies to merge:', df.index,
                                 format_func=lambda x: df['title'].loc[x])

# Check if more than two movies are selected
if len(selected_indices) > 2:
    st.sidebar.warning('Please select at most two movies.')
    # Limit the selected indices to the first two selected indices
    selected_indices = selected_indices[:2]

# Main body - Merge Synopses button and display area
if st.sidebar.button('Merge Synopses') and len(selected_indices) == 2:
    title_1, title_2 = df['title'].loc[selected_indices].values
    title_1, title_2 = title_1.title(), title_2.title()

    syn_1, syn_2 = df['plot_synopsis'].loc[selected_indices].values

    prompt = f"Merge and rewrite the synopsis from '{title_1}' and '{title_2}'. Create a new synopsis incorporating elements from both."

    # Call OpenAI API for synopsis merging
    img, response = openai_api(title_1, title_2, prompt)

    st.title('Generated Movie Poster')
    st.image(img)

    # Display merged synopsis
    merged_synopsis = response.choices[0].message.content  # Ensure correct attribute access based on OpenAI's response structure

    st.title('Merged Synopsis')
    st.write(merged_synopsis)

    st.title("Original Synopses")
    with st.expander(f"Show/hide {title_1} synopsis"):
        syn_1 = summarizer(syn_1, max_length=130, min_length=30, do_sample=False)
        st.markdown(f"""{syn_1}""")

    with st.expander(f"Show/hide {title_2} synopsis"):
        syn_2 = summarizer(syn_2, max_length=130, min_length=30, do_sample=False)
        st.markdown(f"""{syn_2}""")

    faces_title_1, faces_title_2 = movies_to_analyse(title_1, title_2)

    # Display title_1's characters
    st.title(f"Choose Characters from {title_1}")

    # Display images in a row
    st.markdown("<div style='display:flex;'>", unsafe_allow_html=True)
    for image_char in faces_title_1:
        # Extract the NumPy array representing the image data
        image_data = image_char[0]

        # Convert the NumPy array to a PIL image object
        pil_image = Image.fromarray(image_data)

        # Display the image with a clickable radio button
        st.markdown(f"<label><input type='radio' name='image' value='{image_char}'/>", unsafe_allow_html=True)
        st.image(pil_image, use_column_width=False)
        st.markdown("</label>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


    # Get the user's choice
    selected_image_1 = st.radio("Select your favorite image:", faces_title_1, key=1)

    # Display the selected image
    st.write(f"You selected: {selected_image_1}")

    # Display title_2's characters
    st.title(f"Choose Characters from {title_2}")

    # Display images in a row
    st.markdown("<div style='display:flex;'>", unsafe_allow_html=True)
    for image_char in faces_title_2:
        # Extract the NumPy array representing the image data
        image_data = image_char[0]

        # Convert the NumPy array to a PIL image object
        pil_image = Image.fromarray(image_data)

        # Display the image with a clickable radio button
        st.markdown(f"<label><input type='radio' name='image' value='{image_char}'/>", unsafe_allow_html=True)
        st.image(pil_image, use_column_width=False)
        st.markdown("</label>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Get the user's choice
    selected_image_2 = st.radio("Select your favorite image:", faces_title_2, key=2)

    # Display the selected image
    st.write(f"You selected: {selected_image_2}")
