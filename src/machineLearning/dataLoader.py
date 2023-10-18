import pandas as pd

class DataLoader:
    def __init__(self):
        pass

    def loadSplitDataset(self, pathToCSV, header, labelIdentifier):
        dataset = self.loadDataset(pathToCSV, header)
        features, labels = self.splitData(dataset, labelIdentifier)
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
    def encodelabels(labels):
        labelEncoder = LabelEncoder()
        encodedLabels = labelEncoder.fit_transform(labels)
        return encodedLabels