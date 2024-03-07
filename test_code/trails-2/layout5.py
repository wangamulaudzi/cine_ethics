import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Load dataset
file_path = 'test_data_1.csv'  # Update to your file path
df = pd.read_csv(file_path, sep=';')

# Streamlit layout
st.title('Movie Synopsis Merger 🍿🎬')  # Title
st.write("Discover your own innovative plot!")  # Description

# Select Genre
genre = st.selectbox("Search For Movie Titles/ Genre(s):", ["Action", "Adventure", "Crime", "Family", "Fantasy", "Horror", "Mystery", "Romance", "Scifi", "Thriller"])

# Sidebar for search functionality
st.sidebar.title("Search & Select Movies")
search_query = st.sidebar.text_input('Search movie titles or genre(s)')

# Filter dataframe based on search query
filtered_df = df[
    (df['title'].str.contains(search_query, case=False, na=False)) |
    (df['genre'].str.contains(search_query, case=False, na=False))
] if search_query else df

# Display filtered movies
if not filtered_df.empty:
    st.write("Search Results:")
    for index, row in filtered_df.iterrows():
        st.write(f"**{row['title']}** - Genre: {row['genre']}")
        st.write(f"Synopsis: {row['plot_synopsis']}")
else:
    st.warning("No matching movies found.")

# Advanced search options
with st.sidebar.expander("Advanced Search Options"):
    selected_genre = st.selectbox("Select Genre:", df['genre'].unique())
    selected_year = st.slider("Select Release Year:", min_value=df['Year'].min(), max_value=df['Year'].max())

    advanced_filtered_df = df[
        (df['genre'] == selected_genre) &  # Adjusted column name
        (df['Year'] == selected_year)
    ]

    # Display advanced search results
    if not advanced_filtered_df.empty:
        st.write("Advanced Search Results:")
        st.table(advanced_filtered_df[['title', 'genre', 'plot_synopsis']])  # Adjusted column names
    else:
        st.info("No movies found with the selected genre and release year.")

# Selection of movies in the sidebar
selected_indices = st.sidebar.multiselect('Select movies to merge synopses:', filtered_df.index,
                                          format_func=lambda x: filtered_df['title'].loc[x])

st.image('CinePick.png', caption='CinePick logo')

# Main body - Merge Synopses button and display area
if st.sidebar.button('Merge Synopses') and len(selected_indices) == 2:
    title_1, title_2 = filtered_df['title'].iloc[selected_indices].values
    syn_1, syn_2 = filtered_df['plot_synopsis'].iloc[selected_indices].values
    prompt = f"Merge and rewrite the synopses from '{title_1}' and '{title_2}'. Create a new synopsis incorporating elements from both."

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
