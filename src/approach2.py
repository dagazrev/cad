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
        trainPaths, valPaths = self.utils.getImagePathsFromFolders("train", "val")
        for imagePath in trainPaths:
            image = self.utils.loadImage(imagePath)
            #self.plot.showImage(image)
            preprocessed, mask = self.pre.preprocessApproach2(image)
            features = self.feat.color_features(preprocessed, mask)
            label = self.utils.getLabel(imagePath)
            self.utils.store(imagePath, label, features)
            # Display the original and segmented images
            #cv2.imshow('Original Image', image)
            #cv2.imshow('Segmented Image', preprocessed)

            # Wait for a key press and then close the windows
            cv2.waitKey(0)
            cv2.destroyAllWindows()







if __name__ == "__main__":
    app2 = Approach2()
    app2.run()

