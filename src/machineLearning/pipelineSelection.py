# basic
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# preprocessing and feature selection
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer, SplineTransformer
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE, RFECV, VarianceThreshold
from sklearn.model_selection import StratifiedKFold

# classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

# metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# outliers
from outlierDetection import IQRTransformer, ZScoreTransformer, ModifiedZScoreTransformer

# data loader
from dataLoader import DataLoader


class PipelineSelection:
    def __init__(self, approachName):
        self.resultsDirectory = "features/" + approachName + "/"
        self.crossValidation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def selectPipleline(self, features, labels):
        paramGrid = self.createPipelinesAsParameterGrid()
        pipeline = self.createEmptyPipeline()

        selectedFeatures, selectedFeatureNames = self.selectFeatures(features, labels)
        self.storeSelectedFeatures(selectedFeatureNames)

        ## Perform the grid search 
        gridSearch = GridSearchCV(pipeline, paramGrid, scoring="accuracy", cv=self.crossValidation, n_jobs=-1, error_score="raise")
        gridSearch.fit(selectedFeatures, labels)

        self.storeResultsOf(gridSearch)
        print(f"results computed")
        return 0

    @staticmethod
    def createPipelinesAsParameterGrid():
        outlierTransformers = [
            IQRTransformer(),
            ZScoreTransformer(),
            ModifiedZScoreTransformer()
        ]

        scalers = [
            None,
            StandardScaler(),
            RobustScaler(),
            MinMaxScaler(),
            MaxAbsScaler(),
            QuantileTransformer(n_quantiles=100), 
        ]
        
        classifiers = [
            RandomForestClassifier(),
            GradientBoostingClassifier(),
            SVC(),
            LinearDiscriminantAnalysis(),
            xgb.XGBClassifier(),
            DecisionTreeClassifier()
        ]

        paramGrid = {
            "outlier_Transformer": outlierTransformers,
            "scaler": scalers,
            "classifier": classifiers,
        }

        return paramGrid

    @staticmethod
    def createEmptyPipeline():
        pipeline = Pipeline([
            ("outlier_Transformer", None),
            ("scaler", None),
            ("classifier", None),
        ])
        return pipeline

    def selectFeatures(self, features, labels):
        rfeEstimator = RandomForestClassifier(min_samples_leaf=4, criterion="entropy", n_estimators=100)
        rfecv = RFECV(estimator=rfeEstimator, cv=self.crossValidation, n_jobs=-1)
        rfecv.fit(features, labels)
        selectedFeatureNames = features.columns[rfecv.support_]
        selectedFeatures = features[selectedFeatureNames]
        return selectedFeatures, selectedFeatureNames

    def storeSelectedFeatures(self, selectedFeatureNames):
        featureResultPath = os.path.join(self.resultsDirectory, "selectedFeatures.csv")
        _ = pd.DataFrame(selectedFeatureNames, columns=["selectedFeatures"]).to_csv(featureResultPath, index=False)
        print(f"selected feature Names have been stored in {featureResultPath}")
        return 0

    def storeResultsOf(self, gridSearch):
        resultsFilePath = os.path.join(self.resultsDirectory, "pipelineSelection.csv")
        results = pd.DataFrame(gridSearch.cv_results_)
        results.to_csv(resultsFilePath, index=False)
        return 0



if __name__ == "__main__":
    approachName = "approach1"
    datasetPath = "features/approach1/dataset.csv"

    loader = DataLoader()
    selector = PipelineSelection(approachName)

    features, labels = loader.loadSplitDataset(datasetPath, header=0, labelIdentifier="Label")
    selector.selectPipleline(features, labels)
