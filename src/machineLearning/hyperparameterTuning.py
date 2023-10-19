# basic
import pandas as pd
from sklearn.pipeline import Pipeline
import os
import numpy as np


# outliers
from outlierDetection import IQRTransformer, ZScoreTransformer, ModifiedZScoreTransformer

# scalers and encoders
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler, QuantileTransformer, LabelEncoder, OrdinalEncoder

# classifiers
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier

# gridseach
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# data loader
from dataLoader import DataLoader


class HyperparameterTuner:
    def __init__(self, approachName):
        self.resultsDirectory = "features/" + approachName + "/"
        self.selectFeaturesFilePath = os.path.join(self.resultsDirectory, "selectedFeatures.csv")
        self.pipelineFilePath = os.path.join(self.resultsDirectory, "pipelineSelection.csv")

        self.crossValidation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        self.outlierTransformerLookup = self.initOutlierTransformerLookup()
        self.scalerLookup = self.initScalerLookup()
        self.classifierLookup = self.initClassifierLookup()


    def tuneHyperparameters(self, features, labels):
        selectedFeatureNames = self.readSelectedFeatures(self.selectFeaturesFilePath)
        selectedFeatures = features[selectedFeatureNames]

        # get best pipeline for each classifier
        resultsBestClassifiers = self.getBestPipelines(self.pipelineFilePath)
        bestPipelines = self.initPipelines(resultsBestClassifiers)

        # gridsearch for hyperparameter tuning with best pipeline for each classifier
        results_df = pd.DataFrame(columns=["pipeline", "Classifier", "Best Parameters", "Best Score", "Fold Scores", "standard deviation"])
        for index in bestPipelines:
            pipeline = bestPipelines[index]
            hyperparameterGrid = self.matchClassifierToGrid(pipeline)
            gridSearch = GridSearchCV(pipeline, hyperparameterGrid, cv=self.crossValidation, n_jobs=-1)
            gridSearch.fit(selectedFeatures, labels)
            self.storeResults(gridSearch, pipeline, results_df)

    @staticmethod
    def readSelectedFeatures(selectFeaturesFilePath):
        df = pd.read_csv(selectFeaturesFilePath)
        selectedFeatures = df["selectedFeatures"].tolist()
        return selectedFeatures

    def getBestPipelines(self, resultsFilePath):
        results = pd.read_csv(resultsFilePath)
        results["mean_test_score"] = pd.to_numeric(results["mean_test_score"], errors="coerce")
        resultsBestClassifiers = results.sort_values("mean_test_score", ascending=False).groupby("param_classifier").first().reset_index()
        return resultsBestClassifiers

    def initPipelines(self, resultsBestClassifiers):
        pipelines = {}
        for index, row in resultsBestClassifiers.iterrows():
            outlierTransformerName = row["param_outlier_Transformer"]
            outlierTransformer = self.outlierTransformerLookup[outlierTransformerName]
            outlierTransformerStep = ("outlierTransformer", outlierTransformer)

            scalerName = row["param_scaler"]
            scaler = self.scalerLookup[scalerName]
            scalerStep = ("scaler", scaler)

            classifierName = row["param_classifier"]
            classifier = self.classifierLookup[classifierName]
            classifierStep = ("classifier", classifier)

            pipeline = Pipeline([outlierTransformerStep, scalerStep, classifierStep])
            pipelines[index] = pipeline
        return pipelines

    @staticmethod
    def initOutlierTransformerLookup():
        outlierTransformerLookup = {
            "IQRTransformer()" : IQRTransformer(),
            "ZScoreTransformer()" : ZScoreTransformer(),
            "ModifiedZScoreTransformer()" : ModifiedZScoreTransformer(),
            "" : None
        }
        return outlierTransformerLookup

    @staticmethod
    def initScalerLookup():
        scalerLookup = {
            "": None,
            "StandardScaler()":  StandardScaler(),
            "RobustScaler()" : RobustScaler(),
            "MinMaxScaler()" : MinMaxScaler(),
            "MaxAbsScaler()" : MaxAbsScaler(),
            "QuantileTransformer(n_quantiles=100)" : QuantileTransformer(n_quantiles=100), 
        }
        return scalerLookup

    @staticmethod
    def initClassifierLookup():
        classifierLookup = {
            "DecisionTreeClassifier()" : DecisionTreeClassifier(),
            "GradientBoostingClassifier()" : GradientBoostingClassifier(),
            "LinearDiscriminantAnalysis()" : LinearDiscriminantAnalysis(),
            "RandomForestClassifier()" : RandomForestClassifier(),
            "SVC()" : SVC(),
            "XGBClassifier(base_score=None, booster=None, callbacks=None,\n              colsample_bylevel=None, colsample_bynode=None,\n              colsample_bytree=None, device=None, early_stopping_rounds=None,\n              enable_categorical=False, eval_metric=None, feature_types=None,\n              gamma=None, grow_policy=None, importance_type=None,\n              interaction_constraints=None, learning_rate=None, max_bin=None,\n              max_cat_threshold=None, max_cat_to_onehot=None,\n              max_delta_step=None, max_depth=None, max_leaves=None,\n              min_child_weight=None, missing=nan, monotone_constraints=None,\n              multi_strategy=None, n_estimators=None, n_jobs=None,\n              num_parallel_tree=None, random_state=None, ...)": xgb.XGBClassifier(),
        }
        return classifierLookup

    def initHyperparameterGrids(self):
        hyperparameterGrids = {
            "randomForestGrid": self.initRandomForestGrid(),
            "svmGrid": self.initSVMGrid(),
            "decisionTreeGrid": self.initDecicisionTreeGrid(),
            "xgbGrid": self.initXGBGrid(),
            "gradientBoostingGrid": self.initGradientBoostingGrid(),
            "ldaGrid": self.initLDAGrid(),
        }
        return hyperparameterGrids

    @staticmethod
    def initRandomForestGrid():
        hyperparameterGrid = {
            "classifier__n_estimators"      : [100, 200],
            "classifier__max_depth"         : [10, 20, 50, 100, None],
            "classifier__min_samples_leaf"  : [1, 2, 4],
            "classifier__min_samples_split" : [2, 5, 10],
        }
        return hyperparameterGrid

    @staticmethod
    def initSVMGrid():
        hyperparameterGrid = {
            "classifier__kernel"  : ["linear", "poly", "rbf", "sigmoid"],
            "classifier__C"       : [0.1, 1, 10, 100, 1000],
            "classifier__gamma"   : [1, 1e-1, 1e-2, 1e-3, 1e-4],
        }
        return hyperparameterGrid

    @staticmethod
    def initDecicisionTreeGrid():
        hyperparameterGrid = {
            "classifier__criterion" : ["gini", "entropy"],
            "classifier__max_depth" : [2,5,20, None],
            "classifier__min_samples_leaf" : [1, 5, 10],
            "classifier__min_samples_split" : [2,10,20],
            "classifier__max_leaf_nodes" : [2, 5, 10, 20],
        }
        return hyperparameterGrid

    @staticmethod
    def initXGBGrid():
        hyperparameterGrid = {
            "classifier__criterion__min_child_weight" : [1, 5, 10],
            "classifier__criterion__gamma"              : [0.5, 1, 1.5, 2, 2.5],
            "classifier__criterion__subsample"          : [0.6, 0.8, 1],
            "classifier__criterion__colsample_bytree"   : [0.6, 0.8, 1],
            "classifier__criterion__max_depth"          : [3, 4, 5],
        }
        return hyperparameterGrid

    @staticmethod
    def initGradientBoostingGrid():
        hyperparameterGrid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__learning_rate": [0.1, 0.01, 0.001],
            "classifier__max_depth": [3, 5, 7],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"]
        }
        return hyperparameterGrid

    @staticmethod
    def initLDAGrid():
        hyperparameterGrid = {
            "classifier__solver": ["svd", "lsqr", "eigen"],
        }
        return hyperparameterGrid

    def matchClassifierToGrid(self, pipeline):
        classifierGridLookup = self.initClassifierGridLookup()
        classifier = pipeline.named_steps["classifier"]
        hyperparameterGrid = classifierGridLookup[str(classifier)]
        return hyperparameterGrid

    def initClassifierGridLookup(self):
        hyperparameterGrids = self.initHyperparameterGrids()
        classifierGridLookup = {
            "DecisionTreeClassifier()" : hyperparameterGrids["decisionTreeGrid"],
            "GradientBoostingClassifier()" : hyperparameterGrids["gradientBoostingGrid"],
            "LinearDiscriminantAnalysis()" : hyperparameterGrids["ldaGrid"],
            "RandomForestClassifier()" :hyperparameterGrids["randomForestGrid"],
            "SVC()" : hyperparameterGrids["svmGrid"],
            "XGBClassifier(base_score=None, booster=None, callbacks=None,\n              colsample_bylevel=None, colsample_bynode=None,\n              colsample_bytree=None, early_stopping_rounds=None,\n              enable_categorical=False, eval_metric=None, feature_types=None,\n              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,\n              interaction_constraints=None, learning_rate=None, max_bin=None,\n              max_cat_threshold=None, max_cat_to_onehot=None,\n              max_delta_step=None, max_depth=None, max_leaves=None,\n              min_child_weight=None, missing=nan, monotone_constraints=None,\n              n_estimators=100, n_jobs=None, num_parallel_tree=None,\n              predictor=None, random_state=None, ...)": hyperparameterGrids["xgbGrid"],
        }
        return classifierGridLookup

    def storeResults(self, gridSearch, pipeline, results_df):
        # get results
        bestParams = gridSearch.best_params_
        bestScore = gridSearch.best_score_
        allResults = gridSearch.cv_results_
        bestIndex = allResults["rank_test_score"].argmin()

        foldScores = []
        for i in range(5):
            foldScores.append(allResults[f"split{i}_test_score"][bestIndex])
        
        # store results
        results_df = results_df._append({
            "pipeline" : pipeline, 
            "Classifier": pipeline.named_steps["classifier"],
            "Best Parameters": bestParams,
            "Best Score": bestScore,
            "Fold Scores": foldScores,
            "standard deviation" : np.std(foldScores)
        }, ignore_index=True)

        # store results
        resultsPath = os.path.join(self.resultsDirectory, "tunedParameters.csv")
        results_df.to_csv(resultsPath, index=False)
        print(f"hyperparameters tuned and results stored")



if __name__ == "__main__":
    approachName = "approach1"
    datasetPath = "features/approach1/dataset.csv"

    loader = DataLoader()
    tuner = HyperparameterTuner(approachName)

    features, labels = loader.loadSplitDataset(datasetPath, header=0, labelIdentifier=-1)
    tuner.tuneHyperparameters(features, labels)