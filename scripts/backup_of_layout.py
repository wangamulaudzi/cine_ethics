import streamlit as st
import pandas as pd
import os
import openai

from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Load dataset###############
path = "/Users/macbookpro/code/wangamulaudzi/cine_ethics/raw_data/final_table.csv"
df = pd.read_csv(path, sep=',', header=0)

# Streamlit layout for sidebar genre selection functionality
st.sidebar.title('Movie Synopsis Merger 🍿🎬')  # Title
st.sidebar.caption("Discover your own innovative plot!")  # Description

search_query = st.sidebar.text_input('Search movie title:', key="search_input")

# Select Genre Radio Button
genre = st.sidebar.radio("Choose the movie genre", df['genre'].unique())

# Filter dataframe based on selected genre
filtered_df = df[df['genre'] == genre]

# Display information about the selected movies
if not filtered_df.empty:
    st.write("Filtered Results:")
    for index, row in filtered_df.iterrows():
        st.write(f"**{row['title']}** - Genre: {row['genre']}")
        st.write(f"Synopsis: {row['synopsis']}")

# Placeholder for selected_indices, modify this according to your previous code
selected_indices = []  # Replace this with your actual selected_indices

# Main body - Merge Synopses button and display area
if st.sidebar.button('Merge Synopses') and len(selected_indices) == 2:
    title_1, title_2 = filtered_df['title'].loc[selected_indices].values
    syn_1, syn_2 = filtered_df['synopsis'].loc[selected_indices].values
    prompt = f"Merge and rewrite the synopsis from '{title_1}' and '{title_2}'. Create a new synopsis incorporating elements from both."

    # Call OpenAI API for synopsis merging
    openai.api_key = openai_api_key
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use a suitable OpenAI engine
        prompt=prompt,
        max_tokens=300
    )

    # Display merged synopsis
    merged_synopsis = response.choices[0].text.strip()
    st.write('Merged Synopsis:', merged_synopsis)
