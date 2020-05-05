import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from settings import *
from itertools import chain


def flatten(l):
   return [element for sub in l for element in sub]

def main():
    genuine_scores = pd.read_csv(GENUINE_SCORE_FILENAME).values.tolist()
    forgery_scores = pd.read_csv(FORGERY_SCORE_FILENAME).values.tolist()

    eer_list = pd.read_csv(EER_LIST_FILENAME).values.tolist()
    auc_list = pd.read_csv(AUC_LIST_FILENAME).values.tolist()

    forgery_scores = flatten(forgery_scores)
    genuine_scores = flatten(genuine_scores)
    eer_list = flatten(eer_list)
    auc_list = flatten(auc_list)

    zeros = np.zeros(len(forgery_scores))
    ones  = np.ones(len(genuine_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((forgery_scores, genuine_scores))
    data = {'Score': scores, 'Genuine': y}

    ax = sns.swarmplot(x=y, y=scores)
    plt.show()


if __name__ == "__main__":
    main()