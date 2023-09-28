from preprocessing import Preprocessing
from featureExtraction import FeatureExtraction
from utilities import Utilities

class Approach1:

    pre = Preprocessing()
    feat = FeatureExtraction()
    util = Utilities()

    def __init__(self):
        pass

    def run(self):
        for imagePath in folders:
            image = utils.loadImage(imagePath)
            preprocessed = pre.preprocessApproach1(image)
            features = feat.extractFeaturesApproach1(preprocessed)
            label = utils.getLabel(imagePath)
            utils.store(imagePath, label, features)

