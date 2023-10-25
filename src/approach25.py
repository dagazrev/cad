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

    def run(self):
        trainPaths, valPaths = self.utils.getImagePathsFromFolders("train", "val")
        storePath = "features/approach25/eval.csv"
        for imagePath in valPaths:
            image = self.utils.loadImage(imagePath)
            preprocessed, mask = self.pre.preprocessApproach25(image)
            print(preprocessed.shape)
            features = self.feat.extractFeaturesApproach25(preprocessed, mask)
            label = self.utils.getLabel(imagePath)
            self.utils.store(storePath,imagePath, label, features)

if __name__ == "__main__":
    app1 = Approach25()
    app1.run()

