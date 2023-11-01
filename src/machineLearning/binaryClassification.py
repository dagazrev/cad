####################################################
## Best Performing Binary Classification Pipeline ##
####################################################
from dataLoader import DataLoader
from outlierDetection import ZScoreTransformer

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

import pandas as pd

class BinaryClassifier:
    def __init__(self):
        self.pipeline = self.initPipeline()
        self.selectFeaturesFilePath = "features/approach1/selectedFeatures.csv"

    def initPipeline(self):
        outlierTransformerStep = ("outlierTransformer", ZScoreTransformer())
        scalerStep = ("scaler", RobustScaler())
        classifierStep = ("classifier", RandomForestClassifier())

        pipeline = Pipeline([outlierTransformerStep, scalerStep, classifierStep])
        return pipeline


    def train(self, features, labels):
        selectedFeatureNames = self.readSelectedFeatures(self.selectFeaturesFilePath)
        selectedFeatures = features[selectedFeatureNames]
        scores = cross_val_score(self.pipeline, selectedFeatures, labels, cv=5)
        print(scores)
        self.pipeline.fit(features, labels) # retrain on full data

    def test(self, features, storePath):
        prediction = self.pipeline.predict(features)
        self.saveToCsv(prediction, storePath)

    def saveToCsv(self, prediction, storePath):
        df = pd.DataFrame(prediction, columns=['Value'])
        df.to_csv(storePath, index=False, header=False)



    @staticmethod
    def readSelectedFeatures(selectFeaturesFilePath):
        df = pd.read_csv(selectFeaturesFilePath)
        selectedFeatures = df["selectedFeatures"].tolist()
        return selectedFeatures

if __name__ == "__main__":
    storePath = "binaryPrediction.csv"

    datasetPath = "features/approach1/dataset.csv"
    testPath = "features/approach1/test_sorted.csv"

    loader = DataLoader()
    classifier = BinaryClassifier()

    features, labels = loader.loadSplitDataset(datasetPath, header=0, labelIdentifier="Label")
    testFeatures, _ = loader.loadSplitDataset(testPath, header=0, labelIdentifier="Label")

    classifier.train(features, labels)
    classifier.test(testFeatures, storePath)


