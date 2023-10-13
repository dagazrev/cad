import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class IQRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, adaptPercentile=5):
        self.lowerWhisker = None
        self.upperWhisker = None
        self.adaptPercentile = adaptPercentile

    def fit(self, X, y=None):
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lowerWhisker = q1 - 1.5 * iqr
        self.upperWhisker = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        XTransformed = np.copy(X)
        for i, (lower, upper) in enumerate(zip(self.lowerWhisker, self.upperWhisker)):
            maskLower = XTransformed[:, i] < lower
            maskUpper = XTransformed[:, i] > upper
            boundaryLower = np.percentile(XTransformed[:, i], self.adaptPercentile)
            boundaryUpper = np.percentile(XTransformed[:, i], 100 - self.adaptPercentile)
            diffLower = np.abs(XTransformed[:, i] - boundaryLower)
            diffUpper = np.abs(XTransformed[:, i] - boundaryUpper)
            XTransformed[maskLower, i] = np.where(diffLower[maskLower] <= diffUpper[maskLower], boundaryLower, boundaryUpper)
            XTransformed[maskUpper, i] = np.where(diffUpper[maskUpper] < diffLower[maskUpper], boundaryUpper, boundaryLower)
        return XTransformed


class ZScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, zThreshold=3.0, adaptPercentile=5):
        self.zThreshold = zThreshold
        self.mean = None
        self.std = None
        self.adaptPercentile = adaptPercentile

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1e-10
        return self

    def transform(self, X):
        XTransformed = np.copy(X)
        for i, (mean, std) in enumerate(zip(self.mean, self.std)):
            zScores = (XTransformed[:, i] - mean) / std
            mask = np.abs(zScores) > self.zThreshold
            boundaryLower = np.percentile(XTransformed[:, i], self.adaptPercentile)
            boundaryUpper = np.percentile(XTransformed[:, i], 100 - self.adaptPercentile)
            diffLower = np.abs(XTransformed[:, i] - boundaryLower)
            diffUpper = np.abs(XTransformed[:, i] - boundaryUpper)
            XTransformed[mask, i] = np.where(zScores[mask] > 0, np.where(diffUpper[mask] < diffLower[mask], boundaryUpper, boundaryLower), np.where(diffLower[mask] <= diffUpper[mask], boundaryLower, boundaryUpper))
        return XTransformed


class ModifiedZScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3.5, adaptPercentile=5):
        self.threshold = threshold
        self.median = None
        self.medianAbsoluteDeviation = None
        self.adaptPercentile = adaptPercentile

    def fit(self, X, y=None):
        self.median = np.median(X, axis=0)
        self.medianAbsoluteDeviation = np.median(np.abs(X - self.median), axis=0)
        self.medianAbsoluteDeviation[self.medianAbsoluteDeviation == 0] = 1e-10
        return self

    def transform(self, X):
        XTransformed = np.copy(X)
        for i, (median, mad) in enumerate(zip(self.median, self.medianAbsoluteDeviation)):
            modifiedZScores = 0.6745 * (XTransformed[:, i] - median) / mad
            mask = np.abs(modifiedZScores) > self.threshold
            boundaryLower = np.percentile(XTransformed[:, i], self.adaptPercentile)
            boundaryUpper = np.percentile(XTransformed[:, i], 100 - self.adaptPercentile)
            diffLower = np.abs(XTransformed[:, i] - boundaryLower)
            diffUpper = np.abs(XTransformed[:, i] - boundaryUpper)
            XTransformed[mask, i] = np.where(modifiedZScores[mask] > 0, np.where(diffUpper[mask] < diffLower[mask], boundaryUpper, boundaryLower), np.where(diffLower[mask] <= diffUpper[mask], boundaryLower, boundaryUpper))
        return XTransformed

