# Import statements
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import openai
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import os

#loading credentials
load_dotenv()

def endpoint(df):
    """
    Creates an endpoint for the suggestion's search bar.
    """
    # API for search bar
    app = FastAPI()

    # Enable CORS (Cross-Origin Resource Sharing) to allow requests from Streamlit app
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Endpoint to fetch suggestions
    @app.get("/suggest")
    async def suggest(query: str):
        suggestions = [title for title in df["title"] if query.lower() in title.lower()]
        return suggestions

def openai_api(title_1, title_2):
    """
    Creates an endpoint for the openai API.

    Params:
        title_1: first movie title
        title_2: second movie title
        prompt: openai text to image prompt

    Returns:
        img: merged image from the two titles
        response: response of the API
    """
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    openai.api_key = openai_api_key
    prompt = f"Merge and rewrite the synopsis from '{title_1}' and '{title_2}'. Create a new synopsis incorporating elements from both."
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
    prompt_image = f"movie poster mixing {title_1} and {title_2}"
    response_dalle = openai.Image.create(
                                        model="dall-e-2",
                                        prompt=prompt_image,
                                        size="1024x1024",
                                        quality="standard",
                                        response_format= "b64_json",
                                        n=1,
                                        )

    firstImage = response_dalle.data[0]
    imgData = base64.b64decode(firstImage.b64_json)
    im_file = BytesIO(imgData)  # convert image to file-like object
    img = Image.open(im_file)   # img is now PIL Image object

    return img, response
