import cv2

class Preprocessing:
    def __init__(self):
        pass

    def preprocessApproach1(self, image):
        rescaledImage = self.rescaleImage(image, scalePercent=75)
        return rescaledImage

    def preprocessApproach2(self, image):
        pass

    def hairRemoval(self, image):
        pass
    
    @staticmethod
    def rescaleImage(image, scalePercent):
        width = int(image.shape[1] * scalePercent / 100)
        height = int(image.shape[0] * scalePercent / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized 