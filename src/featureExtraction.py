import cv2 
import numpy as np
from skimage import feature


class FeatureExtraction:
    def __init__(self):
        pass

    def extractFeaturesApproach1(self, image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        #keypointFeatures = self.extractKeypointFeatures(gray)
        lbpFeatures = self.CompletedRobustLocalBinaryPattern(gray)
        return lbpFeatures

    def extractFeaturesApproach2(self, image):
        pass

    def extractColorFeatures(self, image):
        pass

    def extractShapeFeatures(self, image):
        pass

    def extractSizeFeatures(self, image):
        pass
    
    @staticmethod
    def CompletedRobustLocalBinaryPattern(image, radius=1, neighbors=8):
        lbp = feature.local_binary_pattern(image, neighbors, radius, method='uniform')
        crlbp_result = np.zeros((lbp.shape[0]-2, lbp.shape[1]-2))

        for i in range(1, lbp.shape[0]-1):
            for j in range(1, lbp.shape[1]-1):
                center = lbp[i, j]
                neighbors = [lbp[i-1, j-1], lbp[i-1, j], lbp[i-1, j+1], lbp[i, j+1],
                            lbp[i+1, j+1], lbp[i+1, j], lbp[i+1, j-1], lbp[i, j-1]]

                # Check for the condition of CRLBP
                if (max(neighbors) - min(neighbors)) <= 2:
                    crlbp_result[i-1, j-1] = max(neighbors) - min(neighbors)

        return crlbp_result


    @staticmethod
    def extractKeypointFeatures(image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return descriptors

    def excludeMask(self, image, mask):
        pass
    