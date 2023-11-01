from preprocessing import Preprocessing
from featureExtraction import FeatureExtraction
from utilities import Utilities
from ploting import Ploting

class Approach25:

    pre = Preprocessing()
    feat = FeatureExtraction()
    utils = Utilities()
    plot = Ploting

    def __init__(self):
        pass

    #def run(self):
    #    trainPaths, valPaths = self.utils.getImagePathsFromFoldersNb("train_nb", "val_nb")
    #    storePath = "features/approach25/non_binary_val.csv"
    #    for imagePath in valPaths:
    #        image = self.utils.loadImage(imagePath)
    #        preprocessed, mask = self.pre.preprocessApproach25(image)
    #        print(preprocessed.shape)
    #        features = self.feat.extractFeaturesApproach25(preprocessed, mask)
    #        label = self.utils.getLabel(imagePath)
    #        self.utils.store(storePath,imagePath, label, features)

    def run(self):
        testPaths = self.utils.getImagePathsFromTest("test_nb")
        storePath = "features/approach1/multilabel_test_features.csv"
        for imagePath in testPaths:
            image = self.utils.loadImage(imagePath)
            preprocessed = self.pre.preprocessApproach1(image)
            features = self.feat.extractFeaturesApproach1(preprocessed)
            label = self.utils.getLabel(imagePath)
            self.utils.store(storePath,imagePath, label, features)

if __name__ == "__main__":
    app1 = Approach25()
    app1.run()

