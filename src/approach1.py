from preprocessing import Preprocessing
from featureExtraction import FeatureExtraction
from utilities import Utilities
from ploting import Ploting
import os

class Approach1:

    pre = Preprocessing()
    feat = FeatureExtraction()
    utils = Utilities()
    plot = Ploting

    def __init__(self):
        pass

    def run(self):
<<<<<<< HEAD
        trainPaths, valPaths = self.utils.getImagePathsFromFoldersNb("train_nb", "val_nb")
        storePath = "features/approach1/non_binary.csv"
        storePath2 = "features/approach1/non_binary_val.csv"
        for imagePath in trainPaths:
=======
        trainPaths, valPaths = self.utils.getImagePathsFromFolders("train", "val")
        storePath = "features/approach1/validation.csv"
        for imagePath in valPaths:
>>>>>>> f54bc2404dd47f7a73485358cc7c232637ba4fe9
            image = self.utils.loadImage(imagePath)
            preprocessed = self.pre.preprocessApproach1(image)
            features = self.feat.extractFeaturesApproach1(preprocessed)
            label = self.utils.getLabel(imagePath)
            self.utils.store(storePath,imagePath, label, features)
        for imagePath in valPaths:
            image = self.utils.loadImage(imagePath)
            preprocessed = self.pre.preprocessApproach1(image)
            features = self.feat.extractFeaturesApproach1(preprocessed)
            label = self.utils.getLabel(imagePath)
            self.utils.store(storePath2,imagePath, label, features)

    def test(self):
        testFolder = "testX"
        testPaths = [os.path.join(testFolder, file) for file in os.listdir(testFolder)]
        storePath = "features/approach1/test.csv"
        for imagePath in testPaths:
            image = self.utils.loadImage(imagePath)
            preprocessed = self.pre.preprocessApproach1(image)
            features = self.feat.extractFeaturesApproach1(preprocessed)
            label = 0
            self.utils.store(storePath,imagePath, label, features)

if __name__ == "__main__":
    app1 = Approach1()
    app1.run()

