import os
import glob
import cv2 

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

    @staticmethod
    def loadImage(imagePath):
        print(imagePath)
        return cv2.imread(imagePath)

    @staticmethod
    def getLabel(imagePath):
        components = imagePath.split(os.sep)
        label = components[-2]
        return label

