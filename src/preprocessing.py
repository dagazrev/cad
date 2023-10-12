import cv2
import numpy as np
import sys

class Preprocessing:
    def __init__(self):
        pass

    def preprocessApproach1(self, image):
        rescaledImage = self.rescaleImage(image, scalePercent=75)
        return rescaledImage

    def preprocessApproach2(self, image):
        resized = self.resize_img(image)
        hrem = self.hairRemoval(resized)
        return self.extract_melanoma_blob(hrem)
        pass

    def hairRemoval(self, img):
        # Switch to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create 16 linear structuring elements (SEs)
        linearSEs = [cv2.getStructuringElement(cv2.MORPH_RECT, (22, 1))]

        # Perform sum of black hats
        sum_black_hats = np.zeros_like(img_gray, dtype=np.uint16)
        for se in linearSEs:
            blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, se)
            blackhat = blackhat.astype(np.uint16)
            sum_black_hats += blackhat

        # Normalize
        sum_black_hats = cv2.normalize(sum_black_hats, None, 0, 255, cv2.NORM_MINMAX)
        sum_black_hats = sum_black_hats.astype(np.uint8)

        # Otsu binarization
        _, sum_black_hats = cv2.threshold(sum_black_hats, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Prepare mask for inpainting
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
        sum_black_hats = cv2.dilate(sum_black_hats, kernel)

        # Inpainting
        inpainted_image = cv2.inpaint(img, sum_black_hats, inpaintRadius=15, flags=cv2.INPAINT_TELEA)

        return inpainted_image
        pass


    def resize_img(self,img):
        resize_factor = 0.6
        resized_image = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_LINEAR)
        return resized_image
        pass

        def extract_melanoma_blob(self, original_image, k=2):

        # Reshape the image to a 2D array of pixels
        pixels = original_image.reshape((-1, 3))

        pixels = np.float32(pixels)

        # Apply K-Means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Find the label that corresponds to the melanoma (largest cluster)
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        melanoma_label = unique_labels[np.argmax(label_counts)]

        # Create a mask for the melanoma cluster
        mask = (labels == melanoma_label).reshape(original_image.shape[:2])

        # Apply the mask to the original image
        result_image = cv2.bitwise_and(original_image, original_image, mask=cv2.bitwise_not(mask.astype(np.uint8)*255))
        return result_image, mask
