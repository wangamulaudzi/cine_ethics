import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os

# Load environment variables

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Load dataset
file_path = 'test-scripts-sample.csv'  # Update to your file path
df = pd.read_csv(file_path, sep=';')

# Streamlit layout
st.title('Movie Synopsis Merger')
st.sidebar.image('CinePickSmall.png', caption='Cine Pick')

# Sidebar for search functionality
st.sidebar.title("Search & Select Movies")
search_query = st.sidebar.text_input('Search movie titles')

# Filter dataframe based on search query
filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)] if search_query else df

# Selection of movies in the sidebar
selected_indices = st.sidebar.multiselect('Select movies to merge synopses:', filtered_df.index, format_func=lambda x: filtered_df['title'].loc[x])

# Main body - Merge Synopses button and display area
if st.sidebar.button('Merge Synopses') and len(selected_indices) == 2:
    title_1, title_2 = filtered_df['title'].loc[selected_indices].values
    syn_1, syn_2 = filtered_df['plot_synopsis'].loc[selected_indices].values
    prompt = f"Merge and rewrite the synopses from '{title_1}' and '{title_2}'. Create a new synopsis incorporating elements from both."
    #title_prompt = f"Merge and rewrite the titles from '{title_1}' and '{title_2}'. Create a new title incorporating elements from both."

    # Call OpenAI API for synopsis merging
    openai.api_key = openai_api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Update model as necessary
        messages=[
            {"role": "system", "content": "You are a creative writer. Please merge the following movie synopses."},
            {"role": "user", "content": prompt}

        ],
        max_tokens=200
    )

    # Display merged synopsis
    merged_synopsis = response.choices[0].message.content  # Ensure correct attribute access based on OpenAI's response structure

    st.title('Merged Synopsis')
    st.write('your creation:', merged_synopsis)
    st.title(title_1)
    st.write('Original Synopsis :',title_1, syn_1)
    st.title(title_2)
    st.write('Original Synopsis:',title_2, syn_2)


#st.sidebar.image('https://oaidalleapiprodscus.blob.core.windows.net/private/org-1CS5LUNScN841oINWP8rLwQR/user-j597S[…]g=Qrl21ILqHXx7rTSyrnVDfNPOxTNi8NWr5j7KFatDMew%3D', caption='Cine Pick')

st.image(
            "https://oaidalleapiprodscus.blob.core.windows.net/private/org-1CS5LUNScN841oINWP8rLwQR/user-j597S[…]g=Qrl21ILqHXx7rTSyrnVDfNPOxTNi8NWr5j7KFatDMew%3D",
            width=1024, # Manually Adjust the width of the image as per requirement
        )
