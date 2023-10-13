# Possible libraries to use
import requests
from bs4 import BeautifulSoup
import nltk
import gensim
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import sumy
from nltk.translate.bleu_score import sentence_bleu
import nltk.translate.bleu_score as bleu
from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Step 1: Data Collection
'''
TO-DO:

- Collect documents, research papers, etc.
'''


# Step 2: Preprocessing
'''
TO-DO:

- Remove unnecessary 'noise' from the text
- Tokenize the text
- Remove stop words
- Lemmatize the text
- Stem the text
- uncapitilize the text
'''

# Step 3: Text Vectorization
'''
TO-DO:

- Implement TF-IDF/Word Embeddings
- 
'''

# Step 4: Summarization Model Selection
'''
TO-DO:

- Choose between extractive or abstractive summarization
'''

# Step 5: Model Training
'''
TO-DO:

- Train the summarization model selected
'''

# Step 6: Model Evaluation
'''
TO-DO:

- Evaluate model performance using metrics (ROUGE, BLEU, etc.)
'''

# Step 7: Summarization Inference
'''
TO-DO:

- Deploy model for creating a concise summary of text while still keeping its information/meaning
'''
