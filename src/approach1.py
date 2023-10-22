from preprocessing import Preprocessing
from featureExtraction import FeatureExtraction
from utilities import Utilities
from ploting import Ploting

class Approach1:

    pre = Preprocessing()
    feat = FeatureExtraction()
    utils = Utilities()
    plot = Ploting

    def __init__(self):
        pass

    def run(self):
        trainPaths, valPaths = self.utils.getImagePathsFromFolders("train", "val")
        storePath = "features/approach1/dataset.csv"
        for imagePath in trainPaths:
            image = self.utils.loadImage(imagePath)
            preprocessed = self.pre.preprocessApproach1(image)
            features = self.feat.extractFeaturesApproach1(preprocessed)
            label = self.utils.getLabel(imagePath)
            self.utils.store(storePath,imagePath, label, features)

if __name__ == "__main__":
    app1 = Approach1()
    app1.run()

