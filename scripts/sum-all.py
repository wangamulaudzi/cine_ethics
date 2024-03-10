import pandas as pd
from transformers import pipeline

# Initialize the summarization pipeline with the model
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

# Function to filter DataFrame based on search query
def filter_options(df, search_query, max_suggestions=5):
    """
    Creates a suggestions dataframe based on the user's search_query

    Returns:
        DataFrame
    """
    suggestions = df[df['title'].str.contains(search_query, case=False)][:max_suggestions]
    return suggestions['title'].tolist()

def summarize_column(df, column_name):
    # Function to summarize each row in the specified column
    def summarize_text(text):
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']

    # Apply the summarization function to each row in the specified column
    df['Summarized_' + column_name] = df[column_name].apply(summarize_text)
    return df

# Read the CSV file into a DataFrame
df = pd.read_csv('scripts/final_table.csv')

# Assuming 'plot_synopsis' is the column you want to summarize
df = summarize_column(df, 'plot_synopsis')

# Display the DataFrame with the new column
print(df)
