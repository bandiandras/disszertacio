import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from settings import *
from itertools import chain
sys.path.append(r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\Sig")

from Utils.utils import *


def flatten(l):
   return [element for sub in l for element in sub]

def main():
    # print("FILE GENUINE: "+ FILENAME_GENUINE)
    # df = pd.read_csv(FILENAME_GENUINE)
    # df = utils.standardize_rows(df)

    print("FILE FORGERY: "+ FILENAME_FORGERY)
    df = pd.read_csv(FILENAME_FORGERY)
    df = utils.standardize_rows(df)

    data = []

    for val in df.values:
        xcoords = val[0:512]
        ycoords = val[512:1024]
        userid = val[1024]

        idx = 0
        while (idx < len(xcoords) - 1):
            data.append([xcoords[idx], ycoords[idx], userid])
            idx += 1
    
    df_plot = pd.DataFrame(data, columns=['X1', 'Y1', 'User'])

    df_plot = df_plot.head(51200)

    # sns.pairplot(df_plot,hue='User',size=5,vars=['X1','Y1'],kind='reg')
    sns.lmplot(x="X1", y="Y1", hue="User", data=df_plot)
    plt.title('X-Y Speed user wise - Forgery signatures')
    plt.show()


if __name__ == "__main__":
    main()