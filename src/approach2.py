from preprocessing import Preprocessing
from featureExtraction import FeatureExtraction
from utilities import Utilities
from ploting import Ploting

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
            self.plot.showImage(image)
            preprocessed = self.pre.preprocessApproach1(image)
            features = self.feat.extractFeaturesApproach1(preprocessed)
            label = self.utils.getLabel(imagePath)
            #self.utils.store(imagePath, label, features)

if __name__ == "__main__":
    app1 = Approach2()
    app1.run()

