import os
from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from pandas import read_csv

class utils:

    def getListOfFiles(self, dirName):
        # create a list of file and sub directories 
        # names in the given directory 
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory 
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath)
            else:
                if fullPath.endswith('.csv'):
                    allFiles.append(fullPath)
                    
        return allFiles
    
    @staticmethod
    def readCSVToDataframe(filepath):
       dataframe = read_csv(filepath)
       return dataframe.values


    @staticmethod
    def load_files(folderPath):
        loaded = list()
        filenames = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
        for name in filenames:
            data = utils.readCSVToDataframe(folderPath + "\\" + name)
            loaded.append(data)
        return loaded


    