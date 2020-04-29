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

def returnTraindAndTestDataGenuine(df, userids, i):
    userid = userids[i]
    user_train_data = df.loc[df.iloc[:, -1].isin([userid])]
    # Select data for training
    user_train_data = user_train_data.drop(user_train_data.columns[-1], axis=1)
    user_array = user_train_data.values

    num_samples = user_array.shape[0]
    train_samples = (int)(num_samples * 0.66)
    test_samples = num_samples - train_samples
    # print("#train_samples: "+str(train_samples)+"\t#test_samples: "+ str(test_samples))
    user_train = user_array[0:train_samples,:]
    user_test = user_array[train_samples:num_samples,:]

    other_users_data = df.loc[~df.iloc[:, -1].isin([userid])]
    other_users_data = other_users_data.drop(other_users_data.columns[-1], axis=1)
    other_users_array = other_users_data.values 

    return user_train, user_test, other_users_array 


def returnTraindAndTestDataForgery(df, df_f, userids, i):
    userid = userids[i]
    user_train_data = df.loc[df.iloc[:, -1].isin([userid])]
    # Select data for training
    user_train_data = user_train_data.drop(user_train_data.columns[-1], axis=1)
    user_array = user_train_data.values

    num_samples = user_array.shape[0]
    train_samples = (int)(num_samples * 0.66)
    test_samples = num_samples - train_samples
    # print("#train_samples: "+str(train_samples)+"\t#test_samples: "+ str(test_samples))
    user_train = user_array[0:train_samples,:]
    user_test = user_array[train_samples:num_samples,:]

    other_users_data = df_f.loc[df_f.iloc[:, -1].isin([userid])]
    other_users_data = other_users_data.drop(other_users_data.columns[-1], axis=1)
    other_users_array = other_users_data.values 

    return user_train, user_test, other_users_array 

def main():
    auc_list = []

    print("FILE GENUINE: "+ FILENAME_GENUINE)
    df = pd.read_csv(FILENAME_GENUINE)
    df = utils.standardize_rows(df)

    print("FILE FORGERY: "+ FILENAME_FORGERY)
    df_f = pd.read_csv(FILENAME_FORGERY)
    df_f = utils.standardize_rows(df_f)


    userids = utils.create_userids( df )
    NUM_USERS = len(userids)

    auc_list = list()
    eer_list = list()
    global_positive_scores = list()
    global_negative_scores = list()

    for i in range(0,NUM_USERS):
        if SKILLED_FORGERY:
            user_train, user_test, other_users_array = returnTraindAndTestDataForgery(df, df_f, userids, i)
        else:
            user_train, user_test, other_users_array = returnTraindAndTestDataGenuine(df, userids, i)

        auc = OneClassSVMClassifier(user_train, user_test, other_users_array)
        auc_list.append(auc)

    print('mean: %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )


if __name__ == "__main__":
    main()