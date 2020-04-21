import sys
import os
import random
sys.path.append(os.path.abspath("Model/"))

from settings import *
from utils import *
from Model.point import Point
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

#returns a list of files from the directory given in the dirName parameter (files in nested directories as well) 
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
			allFiles = allFiles + getListOfFiles(fullPath)
		else:
			if fullPath.endswith('.csv'):
				allFiles.append(fullPath)				
	return allFiles


#read CSV file to dataframe and return the values
def readCSVToDataframe(filepath):
	dataframe = read_csv(filepath)
	return dataframe.values


def returnArraysOfXandY(dataframe):
	xCoords = list()
	yCoords = list()
	for entry in dataframe:
		xCoords.append(entry.x)
		yCoords.append(entry.y)
	return np.asarray(xCoords), np.asarray(yCoords)	


def returnStructureOfXYP(dataframe):
	sig = list()
	for entry in dataframe:
		point = Point()
		point.x = entry[0]
		point.y = entry[1]
		point.p = entry[2]
		sig.append(point)
	return sig

	
def resampleSignature(sig, targetLength):
	if (len(sig) > targetLength):
		sig = downSampleSignature(sig, targetLength)
	else:
		sig = upSampleSignature(sig, targetLength)
	return sig


#upsample: insert zeros at the end
#downsample: truncate data
def resampleSignature2(sig, targetLength):
	if (len(sig) > targetLength):
		sig = truncateSignature(sig, targetLength)
	else:
		sig = insertZeros(sig, targetLength)
	return sig


# insert coordinates based on previous and next element, until the desired length is reached
def insertElements (sig, targetLength):
	while (len(sig) < targetLength):
		for i in (1, len(sig) - 1):
			newX = (sig[i-1].x + sig[i].x)/2
			newY = (sig[i-1].y + sig[i].y)/2
			newP = (sig[i-1].p + sig[i].p)/2

			newPoint = Point()
			newPoint.x = newX
			newPoint.y = newY
			newPoint.p = newP
			sig = np.insert(sig, i, newPoint)

			if(len(sig)== targetLength):
				break
	return sig


def upSampleSignature(sig, targetLength):
	sig = insertElements(sig, targetLength)
	return sig


def insertZeros(sig, targetLength):
	while (len(sig) < targetLength):
		p = Point()
		p.x = 0 
		p.y = 0
		p.p = 0
		sig.append(p)
	return sig

def truncateSignature(sig, targetLength):
	return sig[0 : targetLength]

#check actual and target length, accordgin to that, remove every n-th element of the array, until targetLength is reached
#determine n
#possible improvement: multiple smaller downsampleings
def downSampleSignatureOld(sig, targetLength):
	while len(sig) > targetLength:
		i = 1
		while i < len(sig):
			if (len(sig) > targetLength - 2):
				del sig[i]
				i = i + 2
			else:
				break
	return sig 

def downSampleSignature(sig, targetLength):
	while len(sig) > targetLength:
		del sig[random.randint(0, len(sig)-1)]
	return sig 

def main():
	# Get the list of all files in directory tree at given path
	listOfFiles = getListOfFiles(DATASET_PATH)

	loaded = list()

	for name in listOfFiles:
		splittedFilename = name.split('\\')
		newfoldername = NEW_DATASET_PATH + splittedFilename[len(splittedFilename) - 2]
		newFilename = NEW_DATASET_PATH + splittedFilename[len(splittedFilename) - 2] + '\\' + splittedFilename[len(splittedFilename) - 1]
		
		if not os.path.isdir(newfoldername):
			os.makedirs(newfoldername)
		
		with open( newFilename, 'w', newline='') as file:
			writer = csv.writer(file)     
			writer.writerow(['x', 'y', 'p'])
			data = readCSVToDataframe(name)		
			xyp = returnStructureOfXYP(data)
			
			xyp = resampleSignature(xyp, 512)
			# xyp = resampleSignature2(xyp, N)
			for point in xyp:
				writer.writerow([point.x, point.y, point.p])


if __name__ == "__main__":
	main()