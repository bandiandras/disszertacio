import os
from os import listdir
from os.path import isfile, join
import csv
import numpy as np
from pandas import read_csv
from Model.point import Point
from settings import *
from sklearn.preprocessing import MinMaxScaler

class utils:

    #Gets list of files from the directory given in the dirName parameter
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

    @staticmethod
    def normlaizeSignatureMinMax(sig):
        # xCoords = []
        # yCoords = []
        # pList = []
    
        # for point in sig:
        #     xCoords.append(point[0])
        #     yCoords.append(point[1])
        #     pList.append(point[2])
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        # xCoords = scaler.fit_transform(xCoords)
        # yCoords = scaler.fit_transform(yCoords)
        # pList = scaler.fit_transform(pList)

        # cnt = 0
        normalizedSignature = scaler.fit_transform(sig)
        
        # while cnt < len(xCoords) - 1:
        #     point = []
        #     point[0] = xCoords[cnt]
        #     point[1] = yCoords[cnt]
        #     point[2] = pList[cnt]
        #     normalizedSignature.append(point)    
        #     cnt += 1
        
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
    
    @staticmethod
    def writeSpecifiCharacteristicsToCSV(newFilename, data):
        with open( newFilename, 'a', newline='') as file:
            writer = csv.writer(file, dialect='unix', delimiter = ",")
            row = data[0]
            row.extend(data[1])
            row.append(data[2])

            writer.writerow(row)


    