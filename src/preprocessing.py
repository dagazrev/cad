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


    def border_detection(self, image):
        # Define a threshold for black pixel detection
        black_threshold = 10 

        # Check the presence of black pixels in a 1-pixel-wide border
        image_height, image_width, _ = image.shape
        border_pixels = np.sum(image[0:1, :, :] < black_threshold) + \
                        np.sum(image[-1:, :, :] < black_threshold) + \
                        np.sum(image[:, 0:1, :] < black_threshold) + \
                        np.sum(image[:, -1:, :] < black_threshold)

        if border_pixels >= image_width or border_pixels >= image_height:
            # If the black border is detected, proceed with ROI extraction
            # Continue with the code to detect the largest circle and extract the ROI
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            

            # Use Hough Circle Transform to detect circles
            circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, dp=1, minDist=200, param1=50, param2=30, minRadius=200, maxRadius=600)

            circles = np.uint16(np.around(circles))
            largest_circle = max(circles[0], key=lambda circle: circle[2])  # Get the largest circle

            # Create a circular mask
            mask = np.zeros_like(gray_image)
            cv2.circle(mask, (largest_circle[0], largest_circle[1]), largest_circle[2], 255, -1)
                
            # Apply the mask to extract the ROI
            roi = cv2.bitwise_and(image, image, mask=mask)

            result, segmentation_mask = self.extract_melanoma_blob(roi, k=2)
        else:
            result, segmentation_mask = self.extract_melanoma_blob(image, k=2)
        return result, segmentation_mask
