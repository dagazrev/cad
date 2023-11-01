import os
import glob
import cv2 
import csv

class Utilities:
    def __init__(self):
        pass
    
    def getImagePathsFromFolders(self, trainFolder, valFolder):
        trainPaths = self.getRelativePaths(trainFolder)
        valPaths = self.getRelativePaths(valFolder)
        return trainPaths, valPaths

    @staticmethod
    def getRelativePaths(folderPath):
        nevusFolder = os.path.join(folderPath, "nevus")
        othersFolder = os.path.join(folderPath, "others")
        nevusPaths = [os.path.join(nevusFolder, file) for file in os.listdir(nevusFolder)]
        othersPaths = [os.path.join(othersFolder, file) for file in os.listdir(othersFolder)]
        imagePaths = [*nevusPaths, *othersPaths]
        return imagePaths
    
    def getImagePathsFromFoldersNb(self, trainFolder, valFolder):
        trainPaths = self.getRelativePathsNb(trainFolder)
        valPaths = self.getRelativePathsNb(valFolder)
        return trainPaths, valPaths
    
    def getImagePathsFromTest(self, testFolder):
        testPaths = self.getRelativePathsTest(testFolder)
        return testPaths
    
    def getRelativePathsTest(self,folderPath):
        testFolder = os.path.join(folderPath, "test")
        testPaths = [os.path.join(testFolder, file) for file in os.listdir(testFolder)]
        imagePaths = [*testPaths]
        return imagePaths
    
    @staticmethod
    def getRelativePathsNb(folderPath):
        bccFolder = os.path.join(folderPath, "bcc")
        melFolder = os.path.join(folderPath, "mel")
        sccFolder = os.path.join(folderPath, "scc")
        bccPaths = [os.path.join(bccFolder, file) for file in os.listdir(bccFolder)]
        melPaths = [os.path.join(melFolder, file) for file in os.listdir(melFolder)]
        sccPaths = [os.path.join(sccFolder, file) for file in os.listdir(sccFolder)]
        imagePaths = [*bccPaths, *melPaths, *sccPaths]
        return imagePaths

    @staticmethod
    def loadImage(imagePath):
        print(imagePath)
        return cv2.imread(imagePath)

    @staticmethod
    def getLabel(imagePath):
        components = imagePath.split(os.sep)
        label = components[-2]
        return label

    @staticmethod
    def store(filePath, imagePath, label, features):
        csvFile = filePath
        fileExists = os.path.isfile(csvFile)

        if features is None:
            return

        with open(csvFile, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not fileExists:
                writer.writerow(['ImagePath', 'Label'] + [f"Feature_{i}" for i in range(len(features))])

            writer.writerow([imagePath, label] + [str(feature) for feature in features])
