import streamlit as st
import pandas as pd
import openai
import os
from dotenv import load_dotenv

# Configuration and State Management
st.set_page_config(page_title="Movie Synopsis Merger", layout="wide")
st.session_state.setdefault('selected_indices', [])

# Using st.cache_data for data loading
@st.cache_data
def load_data(url):
    data = pd.read_csv(url, sep=';')  # Adjust separator based on your actual data
    return data

# Load API Key from .env
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Validate API Key
if not openai_api_key:
    st.sidebar.error('OpenAI API key not found. Please check your .env file.')
    st.stop()

# Load the dataset
df = load_data("https://github.com/plotly/datasets/raw/master/uber-rides-data1.csv")

# Sidebar for search and selection
st.sidebar.title('Movie Search & Select')
search_query = st.sidebar.text_input('Search movie titles', help='Type to search for movie titles')

# Filter dataset based on search query
if search_query:
    filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)]
else:
    filtered_df = df  # Show all data if no search query

# Display dataframe in main page
st.dataframe(filtered_df)

# Update selected_indices to keep the first selected item if it exists
previous_selection = st.session_state.selected_indices[:1]  # Keep only the first item
options = list(filtered_df.index)
new_selection = st.sidebar.multiselect('Select movies to merge synopses:',
                                       options, default=previous_selection,
                                       format_func=lambda x: filtered_df['title'].loc[x])

# Update the session state
st.session_state.selected_indices = new_selection

st.image('/Users/leonardopacher/code/wangamulaudzi/cine_ethics/notebooks/CinePickSmall.png', caption='Sunrise by the mountains')


# Generate new synopsis button in the sidebar
if st.sidebar.button('Generate Merged Synopsis') and len(st.session_state.selected_indices) == 2:
    selected_df = df.loc[st.session_state.selected_indices]
    titles = selected_df['title'].tolist()
    synopses = selected_df['plot_synopsis'].tolist()

    # Form the prompt for the AI model
    prompt = f"Rewrite the synopsis combining content from {titles[0]} and {titles[1]}, then create a new synopsis from those: {synopses[0]} and {synopses[1]}"

    # Call OpenAI API (ensure you've set your API key in your environment)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Adjust model as needed
        messages=[
            {"role": "system", "content": "You are a creative writer that merges titles and content in a creative way."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    # Display the merged synopsis
    merged_synopsis = response.choices[0].message.content
    st.sidebar.markdown("## Merged Synopsis:")
    st.sidebar.write(merged_synopsis)

# Button to rerun the app
if st.button("Rerun"):
    st.experimental_rerun()
