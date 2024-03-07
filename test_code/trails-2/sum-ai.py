
from openai import OpenAI
import os
from dotenv import load_dotenv


################ the Key ########################
load_dotenv()
#################################################

client = OpenAI()
api_key = os.getenv('OPENAI_API_KEY')


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    #Open Ai AGENT
    {"role": "system", "content": "Create short one paragraph video review"},
    #User
    {"role": "user", "content": "Compose a movie sinopse"}
  ]
)

print(completion.choices[0].message)
