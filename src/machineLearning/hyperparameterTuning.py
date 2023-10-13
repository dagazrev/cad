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



def getBestPipelines(filePath):
    results = pd.read_csv("results/grid_search_results_task01_full_dataset.csv")
    results["mean_test_score"] = pd.to_numeric(results["mean_test_score"], errors="coerce")
    resultsBestClassifiers = results.sort_values("mean_test_score", ascending=False).groupby("param_classifier").first().reset_index()
    return resultsBestClassifiers

def initLookups():
    outlierTransformerLookup = initOutlierTransformerLookup()
    scalerLookup = initScalerLookup()
    classifierLookup = initClassifierLookup()
    return outlierTransformerLookup, scalerLookup, classifierLookup

def initOutlierTransformerLookup():
    outlierTransformerLookup = {
        "IQRTransformer()" : IQRTransformer(),
        "ZScoreTransformer()" : ZScoreTransformer(),
        "ModifiedZScoreTransformer()" : ModifiedZScoreTransformer(),
        "" : None
    }
    return outlierTransformerLookup

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

def initClassifierLookup():
    classifierLookup = {
        "DecisionTreeClassifier()" : DecisionTreeClassifier(),
        "GradientBoostingClassifier()" : GradientBoostingClassifier(),
        "LinearDiscriminantAnalysis()" : LinearDiscriminantAnalysis(),
        "RandomForestClassifier()" : RandomForestClassifier(),
        "SVC()" : SVC(),
        "XGBClassifier(base_score=None, booster=None, callbacks=None,\n              colsample_bylevel=None, colsample_bynode=None,\n              colsample_bytree=None, early_stopping_rounds=None,\n              enable_categorical=False, eval_metric=None, feature_types=None,\n              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,\n              interaction_constraints=None, learning_rate=None, max_bin=None,\n              max_cat_threshold=None, max_cat_to_onehot=None,\n              max_delta_step=None, max_depth=None, max_leaves=None,\n              min_child_weight=None, missing=nan, monotone_constraints=None,\n              n_estimators=100, n_jobs=None, num_parallel_tree=None,\n              predictor=None, random_state=None, ...)": xgb.XGBClassifier(),
    }
    return classifierLookup

def initPipelines(resultsBestClassifiers, outlierTransformerLookup, scalerLookup, classifierLookup):
    pipelines = {}
    for index, row in resultsBestClassifiers.iterrows():
        outlierTransformerName = row["param_outlier_Transformer"]
        outlierTransformer = outlierTransformerLookup[outlierTransformerName]
        outlierTransformerStep = ("outlierTransformer", outlierTransformer)

        scalerName = row["param_scaler"]
        scaler = scalerLookup[scalerName]
        scalerStep = ("scaler", scaler)

        classifierName = row["param_classifier"]
        classifier = classifierLookup[classifierName]
        classifierStep = ("classifier", classifier)

        pipeline = Pipeline([outlierTransformerStep, scalerStep, classifierStep])
        pipelines[index] = pipeline
    return pipelines

def readSelectedFeatures(selectFeaturesFilePath):
    df = pd.read_csv(selectFeaturesFilePath)
    selectedFeatures = df["selectedFeatures"].tolist()
    return selectedFeatures

def loadDataset(fileName, wantedDataset="full"):
    featureDirectory = "Features/"
    dataset = pd.read_csv(featureDirectory + fileName, sep=";", header=0)
    if wantedDataset == "full":
        return dataset
    elif wantedDataset == "air":
        return dataset.drop(dataset.columns[dataset.columns.str.endswith("OP")], axis=1)
    elif wantedDataset == "paper":
        return dataset.drop(dataset.columns[dataset.columns.str.endswith("OA")], axis=1)
    else:
        raise ValueError("Invalid outlier method: " + wantedDataset, + "support datasets are full, air and paper.")
    
def encode(dataset):
    labels = dataset.get("Label")
    features = dataset.drop(["Label","Id"], axis=1)

    # label encoding
    labelEncoder = LabelEncoder()
    encodedLabels = labelEncoder.fit_transform(labels)

    # feature encoding
    categoricalFeatures = ["Sex", "Work"] 
    encoder = OrdinalEncoder()
    features[categoricalFeatures] = encoder.fit_transform(features[categoricalFeatures])
    return features, encodedLabels

def initHyperparameterGrids():
    hyperparameterGrids = {
        "randomForestGrid": initRandomForestGrid(),
        "svmGrid": initSVMGrid(),
        "decisionTreeGrid": initDecicisionTreeGrid(),
        "xgbGrid": initXGBGrid(),
        "gradientBoostingGrid": initGradientBoostingGrid(),
        "ldaGrid": initLDAGrid(),
    }
    return hyperparameterGrids

def initRandomForestGrid():
    hyperparameterGrid = {
        "classifier__n_estimators"      : [100, 200],
        "classifier__max_depth"         : [10, 20, 50, 100, None],
        "classifier__min_samples_leaf"  : [1, 2, 4],
        "classifier__min_samples_split" : [2, 5, 10],
    }
    return hyperparameterGrid

def initSVMGrid():
    hyperparameterGrid = {
        "classifier__kernel"  : ["linear", "poly", "rbf", "sigmoid"],
        "classifier__C"       : [0.1, 1, 10, 100, 1000],
        "classifier__gamma"   : [1, 1e-1, 1e-2, 1e-3, 1e-4],
    }
    return hyperparameterGrid

def initDecicisionTreeGrid():
    hyperparameterGrid = {
        "classifier__criterion" : ["gini", "entropy"],
        "classifier__max_depth" : [2,5,20, None],
        "classifier__min_samples_leaf" : [1, 5, 10],
        "classifier__min_samples_split" : [2,10,20],
        "classifier__max_leaf_nodes" : [2, 5, 10, 20],
    }
    return hyperparameterGrid

def initXGBGrid():
    hyperparameterGrid = {
        "classifier__criterion__min_child_weight" : [1, 5, 10],
        "classifier__criterion__gamma"              : [0.5, 1, 1.5, 2, 2.5],
        "classifier__criterion__subsample"          : [0.6, 0.8, 1],
        "classifier__criterion__colsample_bytree"   : [0.6, 0.8, 1],
        "classifier__criterion__max_depth"          : [3, 4, 5],
    }
    return hyperparameterGrid

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

def initLDAGrid():
    hyperparameterGrid = {
        "classifier__solver": ["svd", "lsqr", "eigen"],
    }
    return hyperparameterGrid

def matchClassifierToGrid(pipeline):
    classifierGridLookup = initClassifierGridLookup()
    classifier = pipeline.named_steps["classifier"]
    hyperparameterGrid = classifierGridLookup[str(classifier)]
    return hyperparameterGrid

def initClassifierGridLookup():
    hyperparameterGrids = initHyperparameterGrids()
    classifierGridLookup = {
        "DecisionTreeClassifier()" : hyperparameterGrids["decisionTreeGrid"],
        "GradientBoostingClassifier()" : hyperparameterGrids["gradientBoostingGrid"],
        "LinearDiscriminantAnalysis()" : hyperparameterGrids["ldaGrid"],
        "RandomForestClassifier()" :hyperparameterGrids["randomForestGrid"],
        "SVC()" : hyperparameterGrids["svmGrid"],
        "XGBClassifier(base_score=None, booster=None, callbacks=None,\n              colsample_bylevel=None, colsample_bynode=None,\n              colsample_bytree=None, early_stopping_rounds=None,\n              enable_categorical=False, eval_metric=None, feature_types=None,\n              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,\n              interaction_constraints=None, learning_rate=None, max_bin=None,\n              max_cat_threshold=None, max_cat_to_onehot=None,\n              max_delta_step=None, max_depth=None, max_leaves=None,\n              min_child_weight=None, missing=nan, monotone_constraints=None,\n              n_estimators=100, n_jobs=None, num_parallel_tree=None,\n              predictor=None, random_state=None, ...)": hyperparameterGrids["xgbGrid"],
    }
    return classifierGridLookup


if __name__ == "__main__":
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    outlierTransformerLookup, scalerLookup, classifierLookup = initLookups()
    folderPath = "./results/results_10_estiamators"

    for fileNumber in ["01", "02", "03", "04", "09", "10"]:
        for wantedDataset in ["full", "air","paper"]:
            # init file names
            fileName = f"AllF_T{fileNumber}.csv"
            selectFeaturesFilePath = os.path.join(folderPath, f"feature_selection_task{fileNumber}_{wantedDataset}_dataset.csv")
            pipelineFilePath = os.path.join(folderPath, f"grid_search_results_task{fileNumber}_{wantedDataset}_dataset.csv")
            
            # load dataset and apply selected features
            dataset = loadDataset(fileName, wantedDataset)
            encodedFeatures, encodedLabels = encode(dataset)
            selectedFeatures = readSelectedFeatures(selectFeaturesFilePath)
            reducedFeatures = encodedFeatures[selectedFeatures]

            # get best pipeline for each classifier
            resultsBestClassifiers = getBestPipelines(pipelineFilePath)
            bestPipelines = initPipelines(resultsBestClassifiers, outlierTransformerLookup, scalerLookup, classifierLookup)

            # gridsearch for hyperparameter tuning with best pipeline for each classifier
            results_df = pd.DataFrame(columns=["pipeline", "Classifier", "Best Parameters", "Best Score", "Fold Scores", "standard deviation"])
            for index in bestPipelines:
                pipeline = bestPipelines[index]
                hyperparameterGrid = matchClassifierToGrid(pipeline)
                gridSearch = GridSearchCV(pipeline, hyperparameterGrid, cv=cv, n_jobs=-1)
                gridSearch.fit(reducedFeatures, encodedLabels)

                # get results
                bestParams = gridSearch.best_params_
                bestScore = gridSearch.best_score_
                allResults = gridSearch.cv_results_
                bestIndex = allResults["rank_test_score"].argmin()

                foldScores = []
                for i in range(5):
                    foldScores.append(allResults[f"split{i}_test_score"][bestIndex])
                
                # store results
                results_df = results_df.append({
                    "pipeline" : pipeline, 
                    "Classifier": pipeline.named_steps["classifier"],
                    "Best Parameters": bestParams,
                    "Best Score": bestScore,
                    "Fold Scores": foldScores,
                    "standard deviation" : np.std(foldScores)
                }, ignore_index=True)

            # store results
            resultsPath = os.path.join(folderPath, f"final_results_task{fileNumber}_{wantedDataset}_dataset.csv")
            results_df.to_csv(resultsPath, index=False)
            print(f"results for task {fileNumber}, {wantedDataset}dataset stored")


