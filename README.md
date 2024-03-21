# CinePick 🍿🎥

A fun Streamlit web app that uses generative AI to create a new synopsis and poster based on two movies chosen by the user.

The web app can be found [here](https://cinepick-front.streamlit.app/).

# Functionality
**Data Consolidation**: Employed Pandas to consolidate data from hundreds of movies into a single table, streamlining the merging and cleaning processes for efficiency and accuracy.

**Synopsis and Poster Generation**: Integrated OpenAI's GPT-3.5-turbo to merge synopses of selected movies, resulting in combined movie descriptions for 250 movies. Utilized DALL-E to generate combined movie posters based on two movie titles.

Facial Recognition: Utilized DLib for facial recognition to identify faces in 1000 screengrabs per movie, enhancing the application's capabilities in character identification.

Character Extraction: Utilized DBSCAN clustering to extract unique characters from identified faces, providing users with a comprehensive overview of movie characters.

Custom Character Merging: Implemented the Hugging Face image-mixer API, enabling users to merge selected characters from different movies seamlessly, enhancing user engagement and customization options.

# Running From Your Terminal
After cloning the repo, simply run the following commands

```
pip install requirements.txt
streamlit run layout.py
```

This will open a streamlit window in your browser to enable you to test the features you have implemented.
