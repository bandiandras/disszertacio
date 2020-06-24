import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import matplotlib.pyplot as plt
import sys
sys.path.append(r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\Sig")

from settings import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from Utils.utils import *


def OneClassSVMClassifierCalculateAUCEER(positive_scores, negative_scores):
    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
 
    scores = np.concatenate((negative_scores, positive_scores))

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    EER = compute_AUC_EER(positive_scores, negative_scores)

    return auc, EER


def OneClassSVMClassifierCalculateScores(user_train, user_test, other_users_array):
    clf = OneClassSVM(gamma='scale')
    clf.fit(user_train)

    positive_scores = clf.score_samples(user_test)
    negative_scores =  clf.score_samples(other_users_array)   
    
    return positive_scores, negative_scores

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

def split_data(df) :
    train_data = df.iloc[:int(df.shape[0] / 3)]
    test_data = df.iloc[int(df.shape[0] / 3):]
    return train_data, test_data

def compute_AUC_EER(positive_scores, negative_scores):  
    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, threshold = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # Calculating EER
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    return EER

def get_model_output_features( df ):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    X = X.reshape(-1, FEATURES, DIMENSIONS)

    model_path = MODEL_PATH
    model = tf.keras.models.load_model(model_path)
    model._layers.pop()
    model.outputs = [model.layers[-1].output]

    X = np.asarray(X).astype(np.float32)

    features = model.predict( X )
    df = pd.DataFrame( features )
    df['user'] = y 
    df.to_csv('features.csv', header = False, index=False)  
    return df

#return forgery features as well
def fine_tune_mode_and_get_model_output_features( df, df_f = None ):
    df_train, df_test = split_data(df)

    y = df_train[df_train.columns[-1]].values
    y = LabelEncoder().fit_transform(y) + 1
    X = df_train.drop(df_train.columns[-1], axis=1).values

    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    X = X.reshape(-1, 128, 8)

    #add columns of zeros to y
    z = np.zeros((y.shape[0], 100-y.shape[1]))
    y = np.append(y, z, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    mini_batch_size = int(min(X.shape[0]/10, BATCH_SIZE))
    start_time = time.time()

    model = tf.keras.models.load_model(MODEL_PATH)
    hist = model.fit(X_train, y_train, 
                        batch_size=mini_batch_size, 
                        epochs=EPOCHS,
                        verbose=True, 
                        validation_data=(X_val, y_val))
    
    array = df_test.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    X = X.reshape(-1, FEATURES, DIMENSIONS)

    
    model._layers.pop()
    model.outputs = [model.layers[-1].output]

    X = np.asarray(X).astype(np.float32)

    features = model.predict( X )
    df = pd.DataFrame( features )
    df['user'] = y 
    df.to_csv('features.csv', header = False, index=False)  

    if (df_f is not None):
        array = df_f.values
        nsamples, nfeatures = array.shape
        nfeatures = nfeatures -1 
        X = array[:,0:nfeatures]
        y = array[:,-1]
        X = X.reshape(-1, FEATURES, DIMENSIONS)

        X = np.asarray(X).astype(np.float32)

        features_forgery = model.predict( X )
        df_f = pd.DataFrame( features_forgery )
        df_f['user'] = y 
        df_f.to_csv('features_forgery.csv', header = False, index=False)  
    return df, df_test, df_f

def main():
    auc_list = []

    auc_list = list()
    eer_list = list()
    global_positive_scores = list()
    global_negative_scores = list()

    print("FILE GENUINE: "+ FILENAME_GENUINE)
    df = pd.read_csv(FILENAME_GENUINE)
    df = utils.standardize_rows(df)

    print("FILE FORGERY: "+ FILENAME_FORGERY)
    df_f = pd.read_csv(FILENAME_FORGERY)
    df_f = utils.standardize_rows(df_f)

    if (FINE_TUNE):        
        features, df_test, features_forgery =  fine_tune_mode_and_get_model_output_features( df, df_f )
        userids = utils.create_userids( df_test )
        NUM_USERS = len(userids)
        
    else:
        features = get_model_output_features(df)
        userids = utils.create_userids( df )
        NUM_USERS = len(userids)
        features_forgery = get_model_output_features(df_f)

   
    for i in range(0,NUM_USERS):
            if SKILLED_FORGERY:
                user_train, user_test, other_users_array = returnTraindAndTestDataForgery(features, features_forgery, userids, i)
            else:
                user_train, user_test, other_users_array = returnTraindAndTestDataGenuine(features, userids, i)

            positive_scores, negative_scores = OneClassSVMClassifierCalculateScores(user_train, user_test, other_users_array)
            global_positive_scores.extend(positive_scores)
            global_negative_scores.extend(negative_scores)
            auc, eer = OneClassSVMClassifierCalculateAUCEER(positive_scores, negative_scores)
            auc_list.append(auc)
            eer_list.append(eer)

    globalAUC, globalEER = OneClassSVMClassifierCalculateAUCEER(global_positive_scores, global_negative_scores)
    globalAUC, globalEER = OneClassSVMClassifierCalculateAUCEER(global_positive_scores, global_negative_scores)
    print('aAUC mean: %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )
    print('aEER mean: %7.4f, std: %7.4f' % ( np.mean(eer_list), np.std(eer_list)) )
    print('AUC mean: %7.4f' % (globalAUC))
    print('EER mean: %7.4f' % (globalEER))


if __name__ == "__main__":
	main()