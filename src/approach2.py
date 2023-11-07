from preprocessing import Preprocessing
from featureExtraction import FeatureExtraction
from utilities import Utilities
from ploting import Ploting
import cv2
import numpy as np

class Approach2:

    pre = Preprocessing()
    feat = FeatureExtraction()
    utils = Utilities()
    plot = Ploting

    def __init__(self):
        pass

    def run(self):
        trainPaths, valPaths = self.utils.getImagePathsFromFoldersNb("train_nb", "val_nb")
        storePath = "features/approach2/non_binary.csv"
        storePath2 = "features/approach2/non_binary_val.csv"
        for imagePath in trainPaths:
            image = self.utils.loadImage(imagePath)
            preprocessed, mask = self.pre.preprocessApproach2(image)
            features = self.feat.color_features(preprocessed, mask)
            label = self.utils.getLabel(imagePath)
            self.utils.store(storePath,imagePath, label, features)

        for imagePath in valPaths:
            image = self.utils.loadImage(imagePath)
            preprocessed, mask = self.pre.preprocessApproach2(image)
            features = self.feat.color_features(preprocessed, mask)
            label = self.utils.getLabel(imagePath)
            self.utils.store(storePath2,imagePath, label, features)







if __name__ == "__main__":
    app2 = Approach2()
    app2.run()

