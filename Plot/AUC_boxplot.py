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
    auc_list_sf = pd.read_csv(AUC_LIST_SKILLED_FILENAME).values.tolist()
    auc_list_rf = pd.read_csv(AUC_LIST_RANDOM_FILENAME).values.tolist()

    auc_list_sf = flatten(auc_list_sf)
    auc_list_rf = flatten(auc_list_rf)

    zeros = np.zeros(len(auc_list_sf))
    ones  = np.ones(len(auc_list_rf))
    y = np.concatenate((zeros, ones))
    AUC = np.concatenate((auc_list_sf, auc_list_rf))
    data = {'AUC': AUC, 'Skilled': y}

    # plt.plot(scores, y)
    ax = sns.boxplot(
        x="Skilled", 
        y="AUC", 
        hue="Skilled",
        data=data, 
        palette="Set3"
    )

    handles, _ = ax.get_legend_handles_labels()
    ax.set_title('AUC értékek Random és Skilled forgery esetén - MCYT adathamlaz')
    ax.legend(handles, ['AUC - Skilled Forgery', 'AUC - Random Forgery'])
    plt.xlabel('Random/Skilled')
    plt.ylabel('AUC')
    plt.show(ax)
    

if __name__ == "__main__":
    main()