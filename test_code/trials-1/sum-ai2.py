import openai
import os
from dotenv import load_dotenv


################ the Key ########################
load_dotenv()
#################################################

# Set your OpenAI API key here
openai.api_key = os.getenv('OPENAI_API_KEY')

# Example of using the updated OpenAI API for a chat completion
response = openai.Completion.create(
  model="text-davinci-003",  # Or another model name, adjust based on your requirements
  prompt="Translate the following English text to French: '{}'",  # Adjust your prompt accordingly
  temperature=0.5,
  max_tokens=100,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Print out the completion
print(response.choices[0].text.strip())
