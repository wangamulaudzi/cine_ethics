import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv

# Setup
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.sidebar.error('OpenAI API key not found. Please check your .env file.')
    st.stop()

@st.cache_data
def load_data(url):
    return pd.read_csv(url, sep=';')  # Adjust based on your actual data

# Load and sort data
df = load_data("test-scripts-sample.csv")  # Replace with your dataset's path
df_sorted = df.sort_values('title')

# Sidebar
st.sidebar.title('Movie Synopsis Merger')

# Movie selection
selected_indices = st.sidebar.multiselect(
    'Select two movies to merge their synopses:',
    df_sorted.index,
    format_func=lambda x: df_sorted.loc[x, 'title']
)

# Main Body
if len(selected_indices) == 2:
    # Titles and synopses extraction
    selected_titles = df_sorted.loc[selected_indices, 'title'].values
    selected_synopses = df_sorted.loc[selected_indices, 'plot_synopsis'].values

    # Displaying the selected titles and synopses side by side in yellow
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {selected_titles[0]}", unsafe_allow_html=True)
        st.markdown(f"<span style='background-color:yellow'>{selected_synopses[0]}</span>", unsafe_allow_html=True)
    with col2:  
        st.markdown(f"### {selected_titles[1]}", unsafe_allow_html=True)
        st.markdown(f"<span style='background-color:yellow'>{selected_synopses[1]}</span>", unsafe_allow_html=True)
        st.image('sunrise.jpg', caption='Sunrise by the mountains')

    # Button for generating merged synopsis
    if st.sidebar.button('Generate Merged Synopsis'):
        # Creating prompt for the AI model
        prompt = f"Create a new title and synopsis by merging the content of '{selected_titles[0]}' and '{selected_titles[1]}': {selected_synopses[0]} {selected_synopses[1]}"

        # OpenAI API call
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Adjust according to availability
            messages=[
                {"role": "system", "content": "You are a creative writer tasked with merging titles and content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )

        # Display merged synopsis in the main page body
        merged_synopsis = response.choices[0].message.content
        st.write('## Merged Synopsis:', merged_synopsis)

elif len(selected_indices) > 2:
    st.sidebar.warning('Please select exactly two movies.')

# Additional sidebar instructions
st.sidebar.text("Please choose two movies from the list above and click 'Generate Merged Synopsis' to create a new, combined synopsis based on the selected movies.")
