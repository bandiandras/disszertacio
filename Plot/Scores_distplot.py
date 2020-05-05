import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from settings import *
from itertools import chain
from sklearn.preprocessing import MinMaxScaler


def flatten(l):
   return [element for sub in l for element in sub]


def main():
    positive_scores = pd.read_csv(GENUINE_SCORE_RANDOM_FILENAME).values.tolist()
    negative_scores = pd.read_csv(FORGERY_SCORE_RANDOM_FILENAME).values.tolist()
    
    # positive_scores = flatten(positive_scores)
    # negative_scores = flatten(negative_scores)
    scaler = MinMaxScaler(feature_range=(0, 1))

    positive_scores = scaler.fit_transform(positive_scores)
    negative_scores = scaler.fit_transform(negative_scores)

    negative_scores = negative_scores[:len(positive_scores)]

    positive_scores = flatten(positive_scores)
    negative_scores = flatten(negative_scores)
    # zeros = np.zeros(len(negative_scores))
    # ones  = np.ones(len(positive_scores))
    # y = np.concatenate((zeros, ones))
    # scores = np.concatenate((negative_scores, positive_scores))
    # data = {'Scores': scores, 'Genuine': y}
    
    # Histograms for each species
    sns.distplot(negative_scores, label="Negative Scores", kde=False)
    sns.distplot(positive_scores, label="Positive Scores", kde=False)

    # Add title
    plt.title("Distribution of negatve and positive scores returned by OneClass SVM")

    # Force legend to appear
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()