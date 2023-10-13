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


class PipelineSelection:
    def __init__(self):
        paramGrid = self.createPipelinesAsParameterGrid()

    def createPipelinesAsParameterGrid():
        outlierTransformers = [
            IQRTransformer()
            ZScoreTransformer()
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

        selectors = [
            RandomForestClassifier(min_samples_leaf=4, criterion="entropy", n_estimators=400)
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

    def evaluate(features, labels):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        


if __name__ == "__main__":
    folderPath = "./results"
    for fileNumber in ["01", "02", "03", "04", "09", "10"]:
        for wantedDataset in ["full", "air","paper"]:
            fileName = f"AllF_T{fileNumber}.csv"
            dataset = loadDataset(fileName, wantedDataset)
            encodedFeatures, encodedLabels = encode(dataset)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # select features
            rfeEstimator = RandomForestClassifier(min_samples_leaf=4, criterion="entropy", n_estimators=400)
            rfecv = RFECV(estimator=rfeEstimator, cv=cv, n_jobs=-1)
            rfecv.fit(encodedFeatures, encodedLabels)
            selectedFeatures = encodedFeatures.columns[rfecv.support_]
            reducedFeatures = encodedFeatures[selectedFeatures]

            # store results
            featureResultPath = os.path.join(folderPath, f"feature_selection_task{fileNumber}_{wantedDataset}_dataset.csv")
            _ = pd.DataFrame(selectedFeatures, columns=["selectedFeatures"]).to_csv(featureResultPath, index=False)
            print(f"features for {wantedDataset} dataset from task {fileNumber} selected")

            # pipeline selection ,(not for hyperparameter tuning)
            paramGrid = createPipelinesAsParameterGrid()

            # Create the pipeline
            pipeline = Pipeline([
                ("outlier_Transformer", None),
                ("scaler", None),
                ("classifier", None),
            ])

            # Perform the grid search 
            gridSearch = GridSearchCV(pipeline, paramGrid, scoring="accuracy", cv=cv, n_jobs=-1, error_score="raise")
            gridSearch.fit(reducedFeatures, encodedLabels)

            resultPath = os.path.join(folderPath, f"grid_search_results_task{fileNumber}_{wantedDataset}_dataset.csv")
            results = pd.DataFrame(gridSearch.cv_results_)
            results.to_csv(resultPath, index=False)
            print(f"results for {wantedDataset} dataset from task {fileNumber} computed")
