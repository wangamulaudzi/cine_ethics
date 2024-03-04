import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

translate = 'my name is flower and I like the skyes'
# Set your OpenAI API key here
openai.api_key = os.getenv('OPENAI_API_KEY')

# Ensure the API key is actually retrieved
if openai.api_key is None:
    raise ValueError("No OpenAI API key found. Make sure your .env file contains 'OPENAI_API_KEY'.")

# Example of using the updated OpenAI API for a text completion
try:
    response = openai.Completion.create(
        engine="text-davinci-003",  # Adjust the engine if necessary
        prompt="Translate the following English text to French: '{translate}'",  # Insert your text inside the curly braces
        temperature=0.5,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Print out the completion
    print(response.choices[0].text.strip())
except Exception as e:
    print(f"An error occurred: {e}")
