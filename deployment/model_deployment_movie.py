#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import sys
import os

def predict_proba(movie):

    clf = joblib.load(os.path.dirname(__file__) + '/movies.pkl')
    data = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
    vect = CountVectorizer(max_features=1000)
    X = vect.fit_transform(data['title'])
    name = vect.transform([movie]) 

    # Make prediction
    p1 = clf.predict_proba(name)

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add a movie')
        
    else:

        movie = sys.argv[1]

        p1 = predict_proba(movie)
        
        print(movie)
        print('Probabilities: ', p1)
        