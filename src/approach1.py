from preprocessing import Preprocessing
from featureExtraction import FeatureExtraction
from utilities import Utilities

class Approach1:

    pre = Preprocessing()
    feat = FeatureExtraction()
    utils = Utilities()

    def __init__(self):
        pass

    def run(self):
        imagePaths = self.utils.getImagePathsFromFolders("train", "val")
        print(len(imagePaths))
        for imagePath in imagePaths:
            image = self.utils.loadImage(imagePath)
            preprocessed = self.pre.preprocessApproach1(image)
            features = self.feat.extractFeaturesApproach1(preprocessed)
            label = self.utils.getLabel(imagePath)
            self.utils.store(imagePath, label, features)

if __name__ == "__main__":
    app1 = Approach1()
    app1.run()

