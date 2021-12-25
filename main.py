from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from textblob import TextBlob
import nltk
import re
import datetime
import math
import matplotlib.pyplot as plt
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVR
from sklearn.compose import ColumnTransformer
import pickle

from sklearn.pipeline import Pipeline




clf = pickle.load(open('SVRmain.pkl', 'rb'))
vectorizer=pickle.load(open('tfidf.pkl','rb'))
app = Flask(__name__)

train = pd.read_csv('train_file.csv')

scaler = StandardScaler()

encoder = LabelEncoder()

stop = set(stopwords.words('english'))
def clean(text):
  text_token = word_tokenize(text)
  filtered_text = ' '.join([w.lower() for w in text_token if w.lower() not in stop and len(w) > 2])
  filtered_text = filtered_text.replace(r"[^a-zA-Z]+", '')
  text_only = re.sub(r'\b\d+\b', '', filtered_text)
  clean_text = text_only.replace(',', '').replace('.', '').replace(':', '')
  return clean_text
  

@app.route('/')
def home():
    return render_template('home.html')




@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        headline = request.form['Headline']
        source=request.form['Source']
        topic=request.form['Topic']
        headline =headline + ' ' + source + ' ' + topic 
        headline = clean(headline)
        transformed_headline = vectorizer.transform([headline]).toarray()
        print("Transformed: ",transformed_headline) 
        vect = pd.DataFrame(transformed_headline)
        
        print("Vector size: ",len(vect))
        
        polarity_h = TextBlob(str(headline)).sentiment.polarity
        subjectivity_h = TextBlob(str(headline)).sentiment.subjectivity      
        num_words_h =  len(str(headline).split())
        num_unique_words_h = len(set(str(headline).split()))
        num_chars_h =  len(str(headline))
        mean_word_len_h = np.mean([len(w) for w in str(headline).split()])


        cols = [num_words_h, num_unique_words_h, num_chars_h, mean_word_len_h,polarity_h,subjectivity_h]
        standard_features = pd.DataFrame(cols)
        
        print("Standard Features: ",len(standard_features))

       

        cols=np.array(cols)
        cols=scaler.fit_transform(cols.reshape(-1,1))
        df_csr = pd.DataFrame(csr_matrix(cols).todense().transpose())
        final_features = pd.concat([vect, df_csr], axis=1)
       
        my_prediction = clf.predict(final_features)
    return render_template('result.html',my_prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)