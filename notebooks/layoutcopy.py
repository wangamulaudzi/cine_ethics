import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os

# Load caio stuff
import base64
from io import BytesIO
from PIL import Image

from google.cloud import storage

# Load New sumarizer stuff
from transformers import pipeline


# Configuration
bucket_name = 'ornate-lens-411311'
file_path = 'path/to/your/file.tsv'
temp_file_path = 'local_temp_file.tsv'  # Local path for temporary storage
# Initialize client and download file
client = storage.Client()
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(file_path)
blob.download_to_filename(temp_file_path)
# Load the file into a pandas DataFrame
df = pd.read_csv(temp_file_path, sep='\t')
# Now, df contains your data and you can perform any operations on it

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

         # Call OpenAI API for DALL-E 2 Text to image
    #client = OpenAI(api_key=openai_api_key)
    prompt = f"movie poster mixing {title_1} and {title_2}"
    response_dalle = openai.Image.create(
                                        model="dall-e-2",
                                        prompt=prompt,
                                        size="1024x1024",
                                        quality="standard",
                                        response_format= "b64_json",
                                        n=1,
                                        )



    firstImage = response_dalle.data[0]
    imgData = base64.b64decode(firstImage.b64_json)
    im_file = BytesIO(imgData)  # convert image to file-like object
    img = Image.open(im_file)   # img is now PIL Image object
    st.write('Generated Movie Poster')
    st.image(img)




    # Display merged synopsis
    merged_synopsis = response.choices[0].message.content  # Ensure correct attribute access based on OpenAI's response structure

    st.title('Merged Synopsis')
    st.write('your creation:', merged_synopsis)
    st.title(title_1)
    st.write('Original Synopsis :',title_1, syn_1)
    st.title(title_2)
    st.write('Original Synopsis:',title_2, syn_2)


#st.sidebar.image('https://oaidalleapiprodscus.blob.core.windows.net/private/org-1CS5LUNScN841oINWP8rLwQR/user-j597S[…]g=Qrl21ILqHXx7rTSyrnVDfNPOxTNi8NWr5j7KFatDMew%3D', caption='Cine Pick')

# st.image(
#             "https://oaidalleapiprodscus.blob.core.windows.net/private/org-1CS5LUNScN841oINWP8rLwQR/user-j597S[…]g=Qrl21ILqHXx7rTSyrnVDfNPOxTNi8NWr5j7KFatDMew%3D",
#             width=1024, # Manually Adjust the width of the image as per requirement
#         )

from google.cloud import storage
import pandas as pd
# Configuration
bucket_name = 'your-bucket-name'
file_path = 'path/to/your/file.tsv'
temp_file_path = 'local_temp_file.tsv'  # Local path for temporary storage
# Initialize client and download file
client = storage.Client()
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(file_path)
blob.download_to_filename(temp_file_path)
# Load the file into a pandas DataFrame
df = pd.read_csv(temp_file_path, sep='\t')
# Now, df contains your data and you can perform any operations on it



summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


print(summarizer(syn_1, max_length=130, min_length=30, do_sample=False))
