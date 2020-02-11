import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
import string
import pandas as pd
from wordcloud import STOPWORDS

def remove_unwated(text):  
    '''Removes punctuations, urls, mentions and hashtags. Returns the string.'''
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", text).split())
    return re.sub(r'@\w+', '', text)

def remove_stopwords(text):
    stopwords_list = STOPWORDS
    cleaned = [word for word in text.split() if (word not in stopwords_list) and len(word) > 1] 
    return " ".join(cleaned) 

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def keys_by_value(d1, value):
    items = d1.items()
    for item  in items:
        for member in item[1]:
            if member == value:
                key = item[0]
    return  key

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))
        
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    X = pd.DataFrame(X, columns=['x1','x2'])

    color = 'blue'
    X.plot.scatter('x1','x2', c=color, s=0.4)

    plt.xticks([]), plt.yticks([])

    if title is not None:
        plt.title(title)
    
class Clean_Tweets(BaseEstimator, TransformerMixin):
    
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_hashtags(self, input_text):
        return re.sub(r"#\\S+", '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  
        return input_text.translate(trantab)
    
    def remove_emoji(self, text): #ok
        '''By compressing the underscore, the emoji is kept as one word'''
        emojis = re.compile('[\U00010000-\U0010ffff]', flags = re.UNICODE)
        return emojis.sub(r'', text)

    def remove_digits(self, text): #ok
        return re.sub('\d+', '', text)
    
    def to_lower(self, text):  #ok
        return text.lower()
    
    def remove_stopwords(self, text):
        stopwords_list = STOPWORDS
        cleaned = [word for word in text.split() if (word not in stopwords_list) and len(word) > 1] 
        return " ".join(cleaned) 
    
    def stemming(self, text):
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in text.split()]
        return " ".join(stemmed)
    
    def lemmatizer(self, text):
        wn = nltk.WordNetLemmatizer()
        text = [wn.lemmatize(word) for word in text.split()]
        return " ".join(text)
    
    def fit(self, X, y = None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_hashtags).apply(self.remove_urls).apply(self.remove_emoji).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming).apply(self.lemmatizer)        
        return clean_X
    
    
    