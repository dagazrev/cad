import os
import glob

class Utilities:
    def __init__(self):
        pass
    
    @staticmethod
    def getImagePathsFromFolders(folder1Path, folder2Path):
        imagePaths1 = os.listdir(folder1Path + "/nevus")
        imagePaths2 = os.listdir(folder1Path + "/others")
        imagePaths3 = os.listdir(folder2Path + "/nevus")
        imagePaths4 = os.listdir(folder2Path + "/others")
        imagePaths = [*imagePaths1, *imagePaths2, *imagePaths3, *imagePaths4]
        return imagePaths


    @staticmethod
    def loadImage(imagePath):
        pass