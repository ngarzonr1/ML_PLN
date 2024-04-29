#!/usr/bin/python

import pandas as pd
import numpy as np
import joblib
import sys
import os

def predict_price(mileage):

    clf = joblib.load(os.path.dirname(__file__) + '/pricing_clf.pkl') 

    arr = np.array([mileage])
    arr = arr.reshape(1,-1)

    # Make prediction
    p1 = clf.predict_price(arr)[0]

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add a Mileage')
        
    else:

        mil = sys.argv[1]

        p1 = predict_price(mil)
        
        print(mil)
        print('Price predicted: ', p1)
        