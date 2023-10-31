import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

class multiclass:
    def __init__(self, approachName, filename, evalname):
        self.data = None
        self.eval = None
        self.features = None
        self.labels = None
        self.evalfeatures = None
        self.evalLabels = None
        self.selected_features = None
        self.selected_evalfeatures = None

        self.datasetFile = filename
        self.evalsetFile = evalname

        self.class_weights = {0: 0.62, 1: 0.89, 2: 4.5}


    def load_data(self):
        self.data = pd.read_csv(self.datasetFile)
        self.eval = pd.read_csv(self.evalsetFile)

        label_mapping = {'bcc': 1, 'mel': 0, 'scc': 2}
        self.data['Label'] = self.data['Label'].map(label_mapping)
        self.eval['Label'] = self.eval['Label'].map(label_mapping)

        self.features = self.data.iloc[:, 2:] 
        self.labels = self.data['Label']
        self.evalfeatures = self.eval.iloc[:, 2:]
        self.evalLabels = self.eval['Label']

    def select_features(self):
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(self.features, self.labels)
        feature_selector = SelectFromModel(rf_classifier, threshold='median')
        feature_selector.fit(self.features, self.labels)

        # Transform the features to select the most important ones
        self.selected_features = feature_selector.transform(self.features)
        self.selected_evalfeatures = feature_selector.transform(self.evalfeatures)

    def evaluate_classifiers(self):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.selected_features)

        # Define a list of classifiers with default parameters
        classifiers = [
            ("GradientBoosting", GradientBoostingClassifier(random_state=42)),
            ("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()),
            ("RandomForest", RandomForestClassifier(random_state=42)),
            ("SVC", SVC(probability=True, random_state=42)),
            ("LogisticRegression", LogisticRegression(random_state=42))
        ]

        top_classifiers = []
        top_scores = []

        kappa_scorer = make_scorer(cohen_kappa_score)
        for name, classifier in classifiers:
            # Use cross-validation to evaluate each classifier
            scores = cross_val_score(classifier, scaled_features, self.labels, cv=5, scoring=kappa_scorer)
            mean_score = scores.mean()

            top_classifiers.append(name)
            top_scores.append(mean_score)

        # Sort the classifiers by performance
        sorted_classifiers = [x for _, x in sorted(zip(top_scores, top_classifiers), reverse=True)]
        sorted_scores = sorted(top_scores, reverse=True)

        # Return the top three classifiers and their scores
        print(sorted_classifiers[:3], sorted_scores[:3])
        return sorted_classifiers[:3], sorted_scores[:3]
    
    def hyperparameter_tuning(self, top_classifiers):
        # Define a custom scorer using Cohen's Kappa
        kappa_scorer = make_scorer(cohen_kappa_score)

        tuned_classifiers = []
        best_parameters = []

        # Define class weights
        class_weights = {0: 0.62, 1: 0.89, 2: 4.5}

        for classifier_name in top_classifiers:
            classifier_parameter_grids = {
                "GradientBoosting": (GradientBoostingClassifier(), {
                    'classifier__n_estimators': [100, 200, 400],  # Adjust parameters as needed
                    'classifier__max_depth': [None, 10, 20],
                }),
                "LinearDiscriminantAnalysis": (LinearDiscriminantAnalysis(), {
                }),
                "RandomForest": (RandomForestClassifier(), {
                    'classifier__n_estimators': [100, 200, 400, 500],  # Adjust parameters as needed
                    'classifier__max_depth': [None, 10, 20],
                }),
                "SVC": (SVC(), {
                    'classifier__C': [1, 0.1],  # Adjust parameters as needed
                    'classifier__kernel': ['linear', 'rbf', 'poly'],
                }),
                "LogisticRegression": (LogisticRegression(), {
                })
            }

            classifier, param_grid = classifier_parameter_grids.get(classifier_name)

            scaler = StandardScaler()
            pipeline = Pipeline([
                ('scaler', scaler),
                ('classifier', classifier)
            ])

            # using experimental grid search
            grid = HalvingGridSearchCV(pipeline, param_grid, scoring=kappa_scorer, cv=5)
            grid.fit(self.selected_features, self.labels)

            best_classifier = grid.best_estimator_.named_steps['classifier']
            best_params = grid.best_params_

            tuned_classifiers.append(best_classifier)
            best_parameters.append(best_params)
        
        return tuned_classifiers, best_parameters

    
    def evaluate_classifier(self, classifier, params):
        scaler = StandardScaler()
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', classifier.set_params(**params))
        ])

        pipeline.fit(self.selected_features, self.labels)

        predictions = pipeline.predict(self.selected_evalfeatures)

        #Evaluation metrics
        accuracy = accuracy_score(self.evalLabels, predictions)
        f1 = f1_score(self.evalLabels, predictions, average='weighted')
        precision = precision_score(self.evalLabels, predictions, average='weighted')
        recall = recall_score(self.evalLabels, predictions, average='weighted')

        #Calculate ROC curve and AUC for each class
        y_score = pipeline.predict_proba(self.selected_features)
        n_classes = len(np.unique(self.evalLabels))
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.evalLabels == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')

        conf_matrix = confusion_matrix(self.evalLabels, predictions)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.show()
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 score: {f1:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

        ks = cohen_kappa_score(self.evalLabels, predictions)
        print(ks)

        target_names = ['class 0', 'class 1', 'class 2']
        print(classification_report(self.evalLabels, predictions, target_names=target_names))
        return
    

if __name__ == "__main__":
    approachName = "approach25"
    filename = "features/approach25/non_binary.csv"
    evalname = "features/approach25/non_binary_val.csv"

    multiclass_instance = multiclass(approachName,filename,evalname)

    multiclass_instance.load_data()
    multiclass_instance.select_features()
    initial_classifiers, initial_scores = multiclass_instance.evaluate_classifiers()
    tuned, best = multiclass_instance.hyperparameter_tuning(initial_classifiers)
    multiclass_instance.evaluate_classifier(tuned, best)
