#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd

# Load NoLBERT (3-class) sentiment pipeline
sentiment_pipeline = pipeline(
    "text-classification",
    model="Ksu246/nolbert-classifier",
    tokenizer="Ksu246/nolbert-classifier",
    return_all_scores=True,
    batch_size=64,
    truncation=True
)

def score_sentences(sentences):
    """
    Runs NoLBERT sentiment scoring on a list of sentences.
    
    Returns a DataFrame with original sentence and sentiment probabilities.
    """
    results = sentiment_pipeline(sentences)

    # Convert results to a DataFrame
    def flatten_results(results):
        flat = []
        for item in results:
            d = {x['label']: x['score'] for x in item}
            flat.append(d)
        return pd.DataFrame(flat)

    # Create base dataframe
    df = pd.DataFrame({"sentence": sentences})
    sentiment_df = flatten_results(results)

    # Rename columns for clarity
    sentiment_df = sentiment_df.rename(columns={
        "LABEL_0": "NEUTRAL",
        "LABEL_1": "POSITIVE",
        "LABEL_2": "NEGATIVE"
    })

    return pd.concat([df, sentiment_df], axis=1)


# In[4]:


pd.set_option('display.max_colwidth', None)    # Allow full sentence text
pd.set_option('display.width', 150)            # Set wide enough console width
pd.set_option('display.max_columns', None)     # Show all columns


# In[5]:


sentences = [
    "We are optimistic about growth.",
    "Margins will contract due to rising costs.",
    "No major updates from management.",
    "Revenue declined by 12% year-over-year.",
    "The board has approved the quarterly dividend.",
    "While demand remained strong, cost inflation rose.",
    "The company sees challenges in China but expects recovery."
]

df = score_sentences(sentences)
print(df)


# In[ ]:




