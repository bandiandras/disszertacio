import sys
import os
sys.path.append(r"C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\Sig")

from settings import *
from Utils.utils import *
from Model.point import Point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
	# Get the list of all files in directory tree at given path
    listOfFiles = utils.getListOfFiles(INPUT_DATASET_PATH)
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

    #standardize signature
    data = utils.normlaizeSignatureMinMax(data)

    sigaWithCharacteristics = utils.calculateCharacterstics(data)
    selectedCharacteristiclist = utils.selectCharacteristics(sigaWithCharacteristics)

    #check filename if signature is genuine or forgery
    #genuine contains v, forgery contains f
    if "f" in splittedFilename[len(splittedFilename) - 1]:
        forgerySignatures = selectedCharacteristiclist
        forgerySignatures.append(userid)
        utils.writeSpecifiCharacteristicsToCSV(OUTPUT_FILE_PATH_GENUINE, forgerySignatures)
    else:
        geniuneSignatures = selectedCharacteristiclist
        geniuneSignatures.append(userid)
        utils.writeSpecifiCharacteristicsToCSV(OUTPUT_FILE_PATH_FORGERY, geniuneSignatures)


    while cnt < len(listOfFiles):
        geniuneSignatures = []
        forgerySignatures = []

        splittedFilename = listOfFiles[cnt].split('\\')
        splittedFilenamePrevioius = listOfFiles[cnt - 1].split('\\')

        userid = splittedFilename[len(splittedFilename) - 2]

        data = utils.readCSVToArray(listOfFiles[cnt])
        sigaWithCharacteristics = utils.calculateCharacterstics(data)
        selectedCharacteristiclist = utils.selectCharacteristics(sigaWithCharacteristics)
        #check filename if signature is genuine or forgery
        #genuine contains v, forgery contains f
        if "f" in splittedFilename[len(splittedFilename) - 1]:
                forgerySignatures = selectedCharacteristiclist
                forgerySignatures.append(userid)              
                utils.writeSpecifiCharacteristicsToCSV(OUTPUT_FILE_PATH_FORGERY, forgerySignatures)
        else:
                geniuneSignatures = selectedCharacteristiclist
                geniuneSignatures.append(userid)
                utils.writeSpecifiCharacteristicsToCSV(OUTPUT_FILE_PATH_GENUINE, geniuneSignatures)

        cnt += 1       
    

if __name__ == "__main__":
    main()