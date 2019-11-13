import pandas as pd
import numpy as np

dfx = pd.read_csv('C:/Users/JAINY/Downloads/Train/Train.csv')
dfx.head()

x = dfxx[:30000,0]
y = dfxx[:30000,1]

print(x.shape)

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Init Objects
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


def getStemmedReview(review):
    
    review = review.lower()
    review = review.replace("<br /><br />"," ")
    
    #Tokenize
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    
    cleaned_review = ' '.join(stemmed_tokens)
    
    return cleaned_review
    
    x_clean = [getStemmedReview(i) for i in x]

    from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
x_vec = cv.fit_transform(x_clean).toarray()
print(x_vec)
print(x_vec.shape)


