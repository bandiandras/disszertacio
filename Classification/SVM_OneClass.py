import sys
import os
import numpy as np
import pandas as pd
sys.path.append(r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\Sig")

from settings import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_score
from matplotlib.pyplot import plot
from sklearn import metrics
from Utils.utils import *

def OneClassSVMClassifier(user_train, user_test, other_users_array):
    clf = OneClassSVM(gamma='scale')
    clf.fit(user_train)

    positive_scores = clf.score_samples(user_test)
    negative_scores =  clf.score_samples(other_users_array)   

    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
 
    scores = np.concatenate((negative_scores, positive_scores))

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def returnTraindAndTestData(df, df_forgery):
    user_train = []
    user_test = []
    other_users_array = []
    userid = ''
    
    user_train = df.head(TRAINING_SAMPLES_SVM)
    df = df.iloc[TRAINING_SAMPLES_SVM:]

    user_test = df.head(GENUINE_SIGNATURES_SVM)
    df = df.iloc[GENUINE_SIGNATURES_SVM:]

    other_users_array = df_forgery.head(FORGERY_SIGNATURES_SVM)

    userid = user_train.iloc[:,-1].values[0]

    #delete userids from every line
    user_train = user_train[user_train.columns[:-1]]
    user_test = user_test[user_test.columns[:-1]]
    other_users_array = other_users_array[other_users_array.columns[:-1]]

    return user_train, user_test, other_users_array, userid 

def main():
    auc_list = []

    print("FILE GENUINE: "+ FILENAME_GENUINE)
    df = pd.read_csv(FILENAME_GENUINE)
    df = utils.standardize_rows(df)

    print("FILE FORGERY: "+ FILENAME_FORGERY)
    df_f = pd.read_csv(FILENAME_FORGERY)
    df_f = utils.standardize_rows(df)

    #iterate through dataframe and make test datasets per user
    while (df.shape[0] > 0) & (df_f.shape[0] > 0):
        user_gen = df.head(NR_OF_GENUINE_SIGS_OF_USER)
        df = df.iloc[NR_OF_GENUINE_SIGS_OF_USER:]

        user_for = df_f.head(NR_OF_FORGERY_SIGS_OF_USER)
        df_f = df_f.iloc[NR_OF_FORGERY_SIGS_OF_USER:]

        user_train, user_test, other_users_array, userid = returnTraindAndTestData(user_gen, user_for)
        auc = OneClassSVMClassifier(user_train, user_test, other_users_array)

        auc_list.append(auc)

    print('mean: %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )


if __name__ == "__main__":
    main()