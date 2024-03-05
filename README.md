# Cine Ethics

## The Data

1. The first dataset contains 800 movies, each with 1k screengrabs from the movie.
2. The second dataset contains 1234 scripts from different movies.
3. The third dataset contains 14000 synopses from different movies.
4. The fourth dataset contains about 240 movies with AI themes.

## Objectives

1. Supervised learning → Image Classification**:**
    1. Train a deep learning model to classify screengrabs into different categories or labels. For example, you could classify screengrabs based on movie genre, scene type (e.g., action, drama, comedy), or specific objects or characters present in the images. (Anasuya)
2. Unsupervised learning → Character Detection:
    1. Develop an object detection model to identify and localize specific characters within the screengrabs.  (Wanga)
    2. Once the characters are identified, a model can be trained to recognize emotions expressed by characters in the screengrabs. This could involve detecting facial expressions or body language cues to infer emotions such as happiness, sadness, anger, etc.
    3. Merge the characters using generative AI. (Caio)
3. Unsupervised learning → Character Analysis
    - **Option A**: Analyze the roles and characteristics of characters in movie scripts. This could involve identifying main characters, analyzing their traits and behaviors, and exploring relationships between characters based on dialogue interactions.
        - **Cluster Analysis:** You can use clustering algorithms (e.g., K-means clustering, hierarchical clustering) to group similar characters based on features extracted from movie scripts. This can help in identifying clusters of characters with similar roles, traits, or narrative functions.
    - **Option B:** Merge the synopses to create a new synopsis (Leo)
