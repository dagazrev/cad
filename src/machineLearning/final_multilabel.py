import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

class final_multilabel:

    def __init__(self):
        pass

    def predicting(self, train_data, test_data, output_csv):
        self.train = pd.read_csv(train_data)
        self.test = pd.read_csv(test_data)

        label_mapping = {'bcc': 1, 'mel': 0, 'scc': 2}
        self.train['Label'] = self.train['Label'].map(label_mapping)

        self.features = self.train.iloc[:, 2:] 
        self.labels = self.train['Label']
        self.test_data = self.test.iloc[:, 2:]

        # Train a GradientBoostingClassifier
        classifier = GradientBoostingClassifier(max_depth=10, n_estimators=200)
        classifier.fit(self.features, self.labels)

        # Predict classes for the test data
        predicted_classes = classifier.predict(self.test_data)

        # Save the predicted classes to a CSV file without a header
        pd.DataFrame(predicted_classes).to_csv(output_csv, header=False, index=False)

if __name__ == "__main__":
    train_data = "features/approach1/non_binary.csv"
    test_data = "features/approach1/multilabel_test_features.csv"
    app1 = final_multilabel()
    app1.predicting(train_data=train_data, test_data=test_data, output_csv="multilabelPrediction.csv")
    

