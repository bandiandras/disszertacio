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
from sklearn.manifold import TSNE

# Plots t-SNE projections of the genuine and forged signatures
# input_csv/forgery_mcyt_1.csv
# input_csv/genuine_mcyt_1.csv
def plot_tsne_binary():
    df_genuine = pd.read_csv(r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\genuine1_MCYT.csv", header =None)
    df_forgery = pd.read_csv(r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\forgery1_MCYT.csv", header =None)
    
    NUM_USERS = 100
    for user in range(0, NUM_USERS):
        userlist = [ user ]
        df_user_genuine = df_genuine.loc[df_genuine[df_genuine.columns[-1]].isin(userlist)]
        df_user_forgery = df_forgery.loc[df_forgery[df_forgery.columns[-1]].isin(userlist)]
 
        print(df_user_genuine.shape)
        print(df_user_forgery.shape)
 
        G = df_user_genuine.values
        F = df_user_forgery.values
 
        df1 = pd.DataFrame(G[:,0:1024])
        df2 = pd.DataFrame(F[:,0:1024])
 
        df = pd.concat( [df1, df2] )
        
 
        print(df.shape)
 
        y = [0] * 25 + [1] * 25  
        y = LabelEncoder().fit_transform(y)
        X = df.values
        tsne = TSNE(n_components=2, init='random', random_state=41)
        X_2d = tsne.fit_transform(X, y)
 
        # target_ids = np.unique(y)
        for i in [0,1]:
            plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1])
        plt.legend(['Genuine', 'Forgery'])
    
        plt.title("User "+str(user))
        plt.show()

def main():
    plot_tsne_binary()

if __name__ == "__main__":
    main()