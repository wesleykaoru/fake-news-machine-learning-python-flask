#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, url_for, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
import pickle


app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        # Creating ngrams
        counts_test = cv.transform([request.form['text']])

        # Send predict
        return render_template('index.html', data=model.predict(counts_test)[0])
    else:
        return render_template('index.html', data='null')


def loadModel():

    # Load ML model
    with open("model.pkl", 'rb') as file:
        Pickled_LR_Model = pickle.load(file)

    # Load vetorizer (vocabulary)
    with open("vectorizer.pkl", 'rb') as file:
        Pickled_Vc = pickle.load(file)

    return(Pickled_LR_Model, Pickled_Vc)


if __name__ == "__main__":
    model = loadModel()[0]
    cv = loadModel()[1]
    app.run(debug=True)
