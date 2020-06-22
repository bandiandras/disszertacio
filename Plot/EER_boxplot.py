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
    eer_list_sf = pd.read_csv(EER_LIST_SKILLED_FILENAME).values.tolist()
    eer_list_rf = pd.read_csv(EER_LIST_RANDOM_FILENAME).values.tolist()

    eer_list_sf = flatten(eer_list_sf)
    eer_list_rf = flatten(eer_list_rf)

    zeros = np.zeros(len(eer_list_sf))
    ones  = np.ones(len(eer_list_rf))
    y = np.concatenate((zeros, ones))
    EER = np.concatenate((eer_list_sf, eer_list_rf))
    data = {'EER': EER, 'Skilled': y}

    # plt.plot(scores, y)
    ax = sns.boxplot(
        x="Skilled", 
        y="EER", 
        hue="Skilled",
        data=data, 
        palette="Set3"
    )

    handles, _ = ax.get_legend_handles_labels()
    ax.set_title('EER értékek Random és Skilled forgery esetén - MCYT adathamlaz')
    ax.legend(handles, ['EER - Skilled Forgery', 'EER - Random Forgery'])
    plt.xlabel('Random/Skilled')
    plt.ylabel('EER')
    plt.show(ax)
    

if __name__ == "__main__":
    main()