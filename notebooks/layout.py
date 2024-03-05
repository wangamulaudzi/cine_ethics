import streamlit as st
import pandas as pd
import openai

#the Key from .env
import os
from dotenv import load_dotenv

#key

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Load the dataset
file_path = '/Users/leonardopacher/code/wangamulaudzi/cine_ethics/notebooks/test-scripts-sample.csv'  # Replace with your file path
df = pd.read_csv(file_path, sep=';')

# Now, you can access the plot_synopsis column like this:
plot_synopses = df['plot_synopsis']

# If you want to print each synopsis individually:
for index, synopsis in enumerate(plot_synopses):
    print(f"Synopsis {index + 1}: {synopsis}\n")

############################# Layout ##########################################
st.title('Movie Synopsis Merger')

# Search box
search_query = st.text_input('Search movie titles')

# Filter dataset based on search query
filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)] if search_query else df

# Checkboxes for each filtered movie title
selected_indices = st.multiselect('Select movies to merge synopses:', filtered_df.index, format_func=lambda x: filtered_df['title'].loc[x])

# When movies are selected and the 'Merge Synopses' button is clicked
if st.button('Merge Synopses') and len(selected_indices) == 2:  # Ensure exactly two movies are selected
    # Extract titles and synopses
    title_1, title_2 = filtered_df['title'].loc[selected_indices].values
    syn_1, syn_2 = filtered_df['plot_synopsis'].loc[selected_indices].values

    # Form the prompt for the AI model
    prompt = f"rewrite the synopsis from two synopsus mixing content and generating a new title from {title_1} and {title_2} after it create a new synopsis from those synopsis {syn_1} and {syn_2}"

    # Call OpenAI API (ensure you've set your API key in your environment)
    openai.api_key = openai_api_key               #st.secrets["OPENAI_API_KEY"]  # Ensure you've added your OpenAI API key in Streamlit secrets
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",  # Adjust model as needed
        messages=[
            {"role": "system", "content": "You are a creative writer that merge in a creative way titles and content."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    # Display the merged synopsis
    merged_synopsis = response["choices"][0]["message"]["content"]
    st.write('Merged Synopsis:', merged_synopsis)




#############Left Nav#######################

# Sidebar for search and selection
st.sidebar.title('Movie Search & Select')
search_query = st.sidebar.text_input('Search movie titles', help='Type to search for movie titles')

# Filter dataset based on search query
if search_query:
    filtered_df = df[df['title'].str.contains(search_query, case=False, na=False)]
else:
    filtered_df = df  # Show all data if no search query

# Sort the filtered DataFrame alphabetically by 'title'
filtered_df = filtered_df.sort_values('title')

# Create a checkbox for each movie title in alphabetical order
selected_titles = []
for index, row in filtered_df.iterrows():
    # This creates a checkbox for each title; if checked, the title is added to the list of selected_titles
    if st.sidebar.checkbox(f"{row['title']}", key=index):
        selected_titles.append(index)

# Update the session state for selected indices based on checkboxes
# This assumes that selected_titles contains indices from the original DataFrame
st.session_state.selected_indices = selected_titles

# Display selected movies' information (if needed) and generate new synopsis
if len(st.session_state.selected_indices) == 2:
    # Assuming you want to do something with the selected movies, like generating a merged synopsis
    selected_df = df.loc[st.session_state.selected_indices]
    # Your code for handling the selected movies goes here...
