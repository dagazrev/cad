import pandas as pd

class DataLoader:
    def __init__(self):
        pass

    def loadDataset(pathToCSV, header):
        dataset = pd.read_csv(pathToCSV, sep=",", header=header)
        return dataset

    def dropColumnsOf(dataset, columns):
        features = dataset.drop(columns, axis=1)
        return features

    def encodelabels(columnName):
        labels = dataset.get(columnName)
        labelEncoder = LabelEncoder()
        encodedLabels = labelEncoder.fit_transform(labels)
        return encodedLabels