import cv2 

class FeatureExtraction:
    def __init__(self):
        pass

    def extractFeaturesApproach1(self, image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        keypointFeatures = self.extractKeypointFeatures(gray)
        return keypointFeatures

    def extractFeaturesApproach2(self, image):
        pass

    def extractColorFeatures(self, image):
        pass

    def extractShapeFeatures(self, image):
        pass

    def extractSizeFeatures(self, image):
        pass

    @staticmethod
    def extractKeypointFeatures(image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return descriptors

    def excludeMask(self, image, mask):
        pass
    