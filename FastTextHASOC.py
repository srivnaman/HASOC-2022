import pandas as pd
import numpy as np
from glob import glob
import re
import json
import gensim
from gensim.models import FastText
from tqdm import tqdm


from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, LSTM

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import stemmer as hindi_stemmer
import nltk

nltk.download('stopwords')

# importing Hindi and English stopwords provided

english_stopwords = stopwords.words("english")
with open('final_stopwords.txt', encoding = 'utf-8') as f:
    hindi_stopwords = f.readlines()
    for i in range(len(hindi_stopwords)):
        hindi_stopwords[i] = re.sub('\n','',hindi_stopwords[i])
stopwords = english_stopwords + hindi_stopwords
english_stemmer = SnowballStemmer("english")

# Function to clean the Hinglish Tweet

regex_for_english_hindi_emojis="[^a-zA-Z#\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF\u0900-\u097F]"
def clean_tweet(tweet):
    tweet = re.sub(r"@[A-Za-z0-9]+",' ', tweet)
    tweet = re.sub(r"https?://[A-Za-z0-9./]+",' ', tweet)
    tweet = re.sub(regex_for_english_hindi_emojis,' ', tweet)
    tweet = re.sub("RT ", " ", tweet)
    tweet = re.sub("\n", " ", tweet)
    tweet = re.sub(r" +", " ", tweet)
    tokens = []
    for token in tweet.split():
        if token not in stopwords:
            token = snow_stemmer.stem(token)  # Stemmer for English Text
            token = hindi_stemmer.hi_stem(token)   # Stemmer for Hindi text
            tokens.append(token)
    return " ".join(tokens)