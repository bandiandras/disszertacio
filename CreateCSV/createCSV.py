from sklearn.preprocessing import MinMaxScaler
import sys
import os
sys.path.append(r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\Sig")

from settings import *
from Utils.utils import *
from Model.point import Point
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

def main():
    sigPath = r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\MCYT_resampled1"
	# Get the list of all files in directory tree at given path
    listOfFiles = utils.getListOfFiles(sigPath)
    loaded = list()

    #counter used for iteriting throug the list of files
    cnt = 1
    splittedFilename = ''
    splittedFilenamePrevioius = ''
    geniuneSignatures = []
    forgerySignatures = []

    splittedFilename = listOfFiles[cnt].split('\\')
    splittedFilenamePrevioius = listOfFiles[cnt - 1].split('\\')

    userid = splittedFilename[len(splittedFilename) - 2]

    #first signature from the dataset
    data = utils.readCSVToArray(listOfFiles[cnt])
    sigaWithCharacteristics = utils.calculateCharacterstics(data)
    selectedCharacteristiclist = utils.selectCharacteristics(sigaWithCharacteristics)
    #check filename if signature is genuine or forgery
    #genuine contains v, forgery contains f
    if "f" in splittedFilename[len(splittedFilename) - 1]:
        forgerySignatures = selectedCharacteristiclist
        forgerySignatures.append(userid)
    else:
        geniuneSignatures = selectedCharacteristiclist
        geniuneSignatures.append(userid)

    while cnt < len(listOfFiles):
        splittedFilename = listOfFiles[cnt].split('\\')
        splittedFilenamePrevioius = listOfFiles[cnt - 1].split('\\')

        if ((cnt > 0) & (splittedFilename[len(splittedFilename) - 2] == splittedFilenamePrevioius[len(splittedFilename) - 2])):
            userid = splittedFilename[len(splittedFilename) - 2]

            data = utils.readCSVToArray(listOfFiles[cnt])
            sigaWithCharacteristics = utils.calculateCharacterstics(data)
            selectedCharacteristiclist = utils.selectCharacteristics(sigaWithCharacteristics)
            #check filename if signature is genuine or forgery
            #genuine contains v, forgery contains f
            if "f" in splittedFilename[len(splittedFilename) - 1]:
                if len(forgerySignatures) == 0:
                    forgerySignatures = selectedCharacteristiclist
                    forgerySignatures.append(userid)
                else:
                    forgerySignatures[0].extend(selectedCharacteristiclist[0])
                    forgerySignatures[1].extend(selectedCharacteristiclist[1])
            else:
                if len(geniuneSignatures) == 0:
                    geniuneSignatures = selectedCharacteristiclist
                    geniuneSignatures.append(userid)
                else:
                    geniuneSignatures[0].extend(selectedCharacteristiclist[0])
                    geniuneSignatures[1].extend(selectedCharacteristiclist[1])
            cnt += 1
        #signature belongs to anmother user -> write line to the csv files
        else:
            #csv
            # genuine.csv (v): 
            #     sor: N(X1) + N(Y1) + userid --> (2N+1)
            # forgery.csv (f)
            #     sor: N(X1) + N(Y1) + userid --> (2N+1)
            utils.writeSpecifiCharacteristicsToCSV(OUTPUT_FILE_PATH_GENUINE, geniuneSignatures)
            utils.writeSpecifiCharacteristicsToCSV(OUTPUT_FILE_PATH_FORGERY, forgerySignatures)

            geniuneSignatures = []
            forgerySignatures = []

            userid = splittedFilename[len(splittedFilename) - 2]

            data = utils.readCSVToArray(listOfFiles[cnt])
            sigaWithCharacteristics = utils.calculateCharacterstics(data)
            selectedCharacteristiclist = utils.selectCharacteristics(sigaWithCharacteristics)
            #check filename if signature is genuine or forgery
            #genuine contains v, forgery contains f
            if "f" in splittedFilename[len(splittedFilename) - 1]:
                if len(forgerySignatures) == 0:
                    forgerySignatures = selectedCharacteristiclist
                    forgerySignatures.append(userid)
                else:
                    forgerySignatures[0].extend(selectedCharacteristiclist[0])
                    forgerySignatures[1].extend(selectedCharacteristiclist[1])
            else:
                if len(geniuneSignatures) == 0:
                    geniuneSignatures = selectedCharacteristiclist
                    geniuneSignatures.append(userid)
                else:
                    geniuneSignatures[0].extend(selectedCharacteristiclist[0])
                    geniuneSignatures[1].extend(selectedCharacteristiclist[1])
            cnt += 1

        

if __name__ == "__main__":
    main()