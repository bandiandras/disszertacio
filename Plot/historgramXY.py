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
import math as m



def main():
    dirName = r"C:\Users\andra\Documents\Egyetem\Allamvizsga\Adat\MOBISIG"
    
    listOfLenghts = []
    # Get the list of all files in directory tree at given path
    listOfFiles = utils.getListOfFiles(dirName)

    for elem in listOfFiles:
        #check length of file and print length to a file
        with open(elem) as f:
            listOfLenghts.append((sum(1 for line in f)))
    
    bins = np.linspace(m.ceil(min(listOfLenghts)), 
                        m.floor(max(listOfLenghts)),
                        100) # fixed number of bins

    plt.xlim([min(listOfLenghts)-5, max(listOfLenghts)+5])

    plt.hist(listOfLenghts, bins=bins, alpha=0.5)
    plt.title('Aláírások hossza - MOBISIG')
    plt.xlabel('Aláírások hossza')
    plt.ylabel('Aláírások száma')
    plt.show()


if __name__ == "__main__":
    main()

