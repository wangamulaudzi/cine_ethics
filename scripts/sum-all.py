import pandas as pd
from transformers import pipeline


df = pd.read_csv('scripts/final_table.csv')

colum_name = df['plot_synopsis']

# Initialize the summarization pipeline with the model
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

def summarize_column(df, column_name):
    # Function to summarize each row in the specified column
    def summarize_text(text):
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']

    # Apply the summarization function to each row in the specified column
    df['Summarized_' + column_name] = df[column_name].apply(summarize_text)
    return df





# Example usage:
# Assuming df is your DataFrame and 'synopsis' is the column you want to summarize
# df = summarize_column(df, 'synopsis')
