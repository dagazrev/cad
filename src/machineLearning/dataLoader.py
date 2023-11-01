import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self):
        pass

    def loadSplitDataset(self, pathToCSV, header, labelIdentifier):
        dataset = self.loadDataset(pathToCSV, header)
        features, labels = self.splitData(dataset, labelIdentifier)
        features = features.drop(["ImagePath"], axis=1)
        if isinstance(labels[1], str):
            labels = self.encodeLabels(labels)
        return features, labels

    @staticmethod
    def loadDataset(pathToCSV, header):
        dataset = pd.read_csv(pathToCSV, sep=",", header=header)
        return dataset

    def splitData(self, dataset, labelIdentifier):
        if isinstance(labelIdentifier, int):
            return self.splitDataByIndex(dataset, labelIdentifier)
        elif isinstance(labelIdentifier, str):
            return self.splitDataByName(dataset, labelIdentifier)
        else:
            raise TypeError("labelIdentifier must be a string or an integer index.")

    @staticmethod
    def splitDataByIndex(dataset, labelIndex):
        features = dataset.drop(dataset.columns[labelIndex], axis=1)
        labels = dataset.iloc[:, labelIndex]
        return features, labels

    @staticmethod
    def splitDataByName(dataset, labelName):
        features = dataset.drop(labelName, axis=1)
        labels = dataset[labelName]
        return features, labels   

    @staticmethod
    def encodeLabels(labels):
        labels = labels.replace({'nevus': 0, 'others': 1})
        return labels
