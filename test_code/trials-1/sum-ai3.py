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


from google.cloud import storage
import pandas as pd
from io import StringIO

# Initialize a storage client using your credentials
client = storage.Client()

# Define your bucket name and object name (file path in the bucket)
bucket_name = 'cine_ethics'
blob_name = 'data/mpst_full_data.csv'

# Create a bucket object and a blob (file) object
bucket = client.get_bucket(bucket_name)
blob = bucket.blob(blob_name)

# Download the file contents into memory
file_contents = blob.download_as_text()

# Use pandas to read the CSV data from the string
df = pd.read_csv(StringIO(file_contents))

# Now df holds your DataFrame, you can inspect it, manipulate it, etc.
print(df.head())





import pandas as pd

# Specify the path to your CSV file
#file_path = '/Users/leonardopacher/code/wangamulaudzi/cine_ethics/notebooks/test-scripts-sample.csv'
file_path2 = '/Users/leonardopacher/code/wangamulaudzi/cine_ethics/notebooks/test_data_1.csv'
# Read the CSV file

df = pd.read_csv(file_path2, delimiter=';')

# Extract the 'plot_synopsis' column
#plot_synopsis = df['plot_synopsis']
# If you want to see the first few entries of the plot_synopsis column, you can use:
print(df.head['plot_synopsis'][0])

#df.fillna('', inplace=True)



# for column in df.columns:
#     print(f"Column {column}:")
#     print(df[column])
#     print()



# Prompt for the AI model
prompt = "Translate the following English text to French: 'Hello, how are you?'"

# Make a request to the API to generate text
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",  # Use the engine of your choice
    messages = [{"role": "user", "content": prompt}],
    max_tokens = 50
)

print(response["choices"][0]["message"]["content"])


################ Huge Robot ###########################s



from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt')

checkpoint = "google/pegasus-xsum"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

long_text = "This is a very very long text. " * 300


sentences = nltk.tokenize.sent_tokenize(long_text)

# initialize
length = 0
chunk = ""
chunks = []
count = -1
for sentence in sentences:
  count += 1
  combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

  if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
    chunk += sentence + " " # add the sentence to the chunk
    length = combined_length # update the length counter

    # if it is the last sentence
    if count == len(sentences) - 1:
      chunks.append(chunk) # save the chunk

  else:
    chunks.append(chunk) # save the chunk
    # reset
    length = 0
    chunk = ""

    # take care of the overflow sentence
    chunk += sentence + " "
    length = len(tokenizer.tokenize(sentence))

# inputs
inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]

# print summary
for input in inputs:
  output = model.generate(**input)
  print(tokenizer.decode(*output, skip_special_tokens=True))
