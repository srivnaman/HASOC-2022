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

df = pd.read_csv("/content/germanTrain.csv")

tweets = df.text
y = df.label

# y = y.replace(to_replace = ['HOF', 'NOT'], value = [1,0])



def Pipeline(tweets):
  #The function takes an array of strings (tweets) as input

    
# Applying the cleaning function to all the tweets 

  cleaned_tweets = [clean_tweet(tweet) for tweet in tweets]

  #Converting the cleaned text into a tokens using inbuilt function of Gensim. PS: please make sure to import gensim
  PreText = [gensim.utils.simple_preprocess(i) for i in cleaned_tweets ]
  
  #Defining Word2Vec Embedding Model
  embedModel = gensim.models.Word2Vec(
    PreText,
    window = 5,
    min_count = 2
    )
  #Training the Embedding model
  embedModel.train(PreText,total_examples=embedModel.corpus_count , epochs = embedModel.epochs)

  #Creating FastText Embedding Model
  FTTmodel = FastText(PreText,
                      window=5,
                      min_count=2)
  
  
  #Creating an average of embedding for each sentence
  def avg_word2vec(doc):
    return np.mean([embedModel.wv[word] for word in doc if word in embedModel.wv.index_to_key], axis=0)
  
  X1 = []
  for i in tqdm(range(len(PreText))):
    X1.append(avg_word2vec(PreText[i]))
  
  X2 = []
  def avg_FTTvec(doc):
    return np.mean([FTTmodel.wv[word] for word in doc if word in FTTmodel.wv.index_to_key], axis=0)
  
  for i in tqdm(range(len(PreText))):
    X2.append(avg_FTTvec(PreText[i]))
  

  X_FTT = np.array(X2)
  X_w2v = np.array(X1)
  
  return X_w2v ,X_FTT  ,embedModel , FTTmodel


X_w2v ,X_Ftv  , W2vModel , FttModel = Pipeline(tweets)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Ftv,y, test_size=0.2, random_state=42)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from lightgbm import LGBMClassifier
from sklearn.svm import NuSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier



ext = ExtraTreesClassifier()
nSvc = NuSVC()
lda = LinearDiscriminantAnalysis()
rcCV = RidgeClassifierCV()
rc = RidgeClassifier(max_iter=1000)
svc = svm.SVC(kernel='rbf')
ccCV = CalibratedClassifierCV()
xgb = XGBClassifier(max_depth = 15,n_estimators=200)
lr = LogisticRegression(max_iter = 1500)
nc = NearestCentroid()
clf = RandomForestClassifier(n_estimators = 200,max_depth=10) 
lgb = LGBMClassifier(max_depth = 25,n_estimators=200)