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
# import stemmer as hindi_stemmer
import nltk

nltk.download('stopwords')