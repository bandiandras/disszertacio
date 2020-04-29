import os
from os import listdir
from os.path import isfile, join
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
from Model.point import Point
from settings import *
from sklearn.preprocessing import MinMaxScaler

class utils:

    @staticmethod
    # function to get unique values 
    def unique(list1):       
        # insert the list to the set 
        list_set = set(list1) 
        # convert the set to the list 
        unique_list = (list(list_set)) 
        return unique_list

    @staticmethod
    # returns the list of unique classids from the last column
    def create_userids( df ):
        array = df.values
        y = array[:, -1]
        return utils.unique( y )

    #Gets list of files from the directory given in the dirName parameter(nested directories as well)
    @staticmethod
    def getListOfFiles(dirName):
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
                allFiles = allFiles + utils.getListOfFiles(fullPath)
            else:
                if fullPath.endswith('.csv'):
                    allFiles.append(fullPath)
                    
        return allFiles

    #Converts the two dimentsional list to a list of Point obejcts
    @staticmethod
    def returnStructureOfXYP(dataframe):
        sig = list()
        for entry in dataframe:
            point = Point()
            point.x = entry[0]
            point.y = entry[1]
            point.p = entry[2]
            sig.append(point)
        return sig

    #selects and returns the selected signature characteristics based on the setting.py
    @staticmethod
    def selectCharacteristics(sig):
        listToReturn = list()
        sublist1 = list()
        sublist2 = list()
        for element in sig:
            if (TIME_SERIES == TimeSeries.XY):
                sublist1.append(element.x)
                sublist2.append(element.y)
            if (TIME_SERIES == TimeSeries.X1Y1) :
                sublist1.append(element.x1)
                sublist2.append(element.y1)
            if (TIME_SERIES == TimeSeries.X2Y2):
                sublist1.append(element.x2)
                sublist2.append(element.y2)
        
        listToReturn.append(sublist1)
        listToReturn.append(sublist2)
        return listToReturn

    #standardizes rows of the dataframe from the df input parameter
    @staticmethod
    def standardize_rows( df):
        array = df.values
        nsamples, nfeatures = array.shape
        nfeatures = nfeatures - 1
        X = array[:, 0:nfeatures]
        y = array[:, -1]
        
        rows, cols = X.shape
        for i in range(0, rows):
            row = X[i,:]
            mu = np.mean( row )
            sigma = np.std( row )
            if( sigma == 0 ):
                sigma = 0.0001
            X[i,:] = (X[i,:] - mu) / sigma    
        df = pd.DataFrame( X )
        df['user'] = y 
        return df

    #calculates X1, Y1, X2 and Y2 characteristics of a signature
    @staticmethod
    def calculateCharacterstics(sig):
        sigWithCharacteristics = []
        idx = 2
        
        #first element of the new list containing characteristics
        point = Point()
        point.x = sig[1][0]
        point.y = sig[1][1]
        point.p = sig[1][2]
        point.x1 = sig[1][0] - sig[0][0]
        point.y1 = sig[1][1] - sig[0][1]
        point.x2 = 0
        point.y2 = 0
        sigWithCharacteristics.append(point)

        while idx < len(sig):
            previousPoint = sigWithCharacteristics[len(sigWithCharacteristics) - 1]

            point = Point()
            point.x = sig[idx][0]
            point.y = sig[idx][1]
            point.p = sig[idx][2]
            point.x1 = sig[idx][0] - sig[idx-1][0]
            point.y1 = sig[idx][1] - sig[idx-1][1]
            point.x2 = point.x1 - previousPoint.x1
            point.y2 = point.y1 - previousPoint.y1
            sigWithCharacteristics.append(point)
            idx += 1
        
        point = Point()
        point.x = sig[idx - 1][0]
        point.y = sig[idx - 1][1]
        point.p = sig[idx - 1][2]
        point.x1 = sig[idx - 1][0] - sig[idx-2][0]
        point.y1 = sig[idx - 1][1] - sig[idx-2][1]
        point.x2 = point.x1 - previousPoint.x1
        point.y2 = point.y1 - previousPoint.y1
        sigWithCharacteristics.append(point)
        return sigWithCharacteristics

    #Normalize the signature using MinMax normalization
    @staticmethod
    def normlaizeSignatureMinMax(sig):
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalizedSignature = scaler.fit_transform(sig)
        return normalizedSignature

   #reads csv file to dataframe and returns the values
    @staticmethod
    def readCSVToArray(filepath):
       dataframe = read_csv(filepath)
       return dataframe.values

    @staticmethod
    def writeDataFrameToCSV(newFilename, data):
        with open( newFilename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y', 'p'])			
            for line in data:
                writer.writerow([line[0], line[1], line[2]])

    @staticmethod
    def writeArrayOfPointsToCSV(newFilename, data):
        with open( newFilename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y', 'p', 'x1', 'y1', 'x2', 'y2'])			
            for line in data:
                writer.writerow([line.x, line.y, line.p, line.x1, line.y1, line.x2, line.y2])       
    
    #Write selected characterisitcs to csv as follows:
    # row: N(X1) + N(Y1) + userid --> (2N+1)
    @staticmethod
    def writeSpecifiCharacteristicsToCSV(newFilename, data):
        with open( newFilename, 'a', newline='') as file:
            writer = csv.writer(file, dialect='unix', delimiter = ",")
            row = data[0]
            row.extend(data[1])
            row.append(data[2])

            writer.writerow(row)


    