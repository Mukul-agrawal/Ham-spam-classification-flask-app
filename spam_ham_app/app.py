from flask import Flask, render_template, request
import string
import pickle
import nltk
import pandas
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)

def text_process(mess):
    nopunc =[char for char in mess if char not in string.punctuation]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    msg = request.form['msg']
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict_proba([msg])
    result = list(prediction[0]).index(max(list(prediction[0])))
    if result == 0:
        var = '1'
    else:
        var = '0'
    return render_template("index.html", var=var)

if __name__ == '__main__':
    app.run(debug=True)
