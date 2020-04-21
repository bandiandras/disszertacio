import sys
import os
import pandas as pd
import numpy as np
sys.path.append(r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\Sig")

from settings import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def main():
    print("FILE: "+ FILENAME)
    df = pd.read_csv(FILENAME)
    
    # df = standardize_rows(df)

    array = df.values
    nsamples, nfeatures = array.shape 
    nfeatures = nfeatures - 1
    X = array[:, 0:nfeatures]
    y = array[:, -1]
    y = np.nan_to_num(y)
    # scaler = MinMaxScaler(feature_range=(0, 1))

    # X = scaler.fit_transform(X)
    X = np.nan_to_num(X)
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS)   
    scoring = ['accuracy']
    num_folds = 10
    scores = cross_val_score(model , X ,y , cv = NUM_FOLDS)
    for i in range(0,num_folds):
        print('\tFold '+str(i+1)+':' + str(scores[ i ]))
    print("accuracy : %0.4f (%0.4f)" % (scores.mean() , scores.std())) 


def standardize_rows( df):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X = array[:, 0:nfeatures]
    y = array[:, -1]
    
    rows, cols = X.shape
    for i in range(0, rows):
        row = X[i,:]
        mu = np.mean( row )
        sigma = np.std( row )
        if( sigma == 0 ):
            sigma = 0.0001
        X[i,:] = (X[i,:] - mu) / sigma    
    df = pd.DataFrame( X )
    df['user'] = y 
    return df

if __name__ == "__main__":
    main()