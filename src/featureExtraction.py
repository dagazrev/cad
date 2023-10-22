import numpy as np
import cv2
import pandas as pd
from skimage import feature
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from skimage.filters import gabor
from skimage import img_as_ubyte
import csv






class FeatureExtraction:
    def __init__(self):
        pass

    def extractFeaturesApproach1(self, image):
        mask = np.full(image.shape, 255)
        glcmFeatures = self.extract_glcm_features(image)
        lbpFeatures = self.extractLBPFeatures(image, mask)
        histFeatures = self.extract3DHistogram(image, mask)
        features = np.concatenate([glcmFeatures, lbpFeatures, histFeatures])
        return features

    def extractFeaturesApproach2(self, image, mask):
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

    def calculate_cluster_prominence(self,glcm):
        # Calculate cluster prominence
        p = np.indices(glcm.shape)[0]
        q = np.indices(glcm.shape)[1]
        sum_rows = np.sum(glcm, axis=1)
        sum_cols = np.sum(glcm, axis=0)
        mean_row = np.sum(p * glcm) / np.sum(glcm)
        mean_col = np.sum(q * glcm) / np.sum(glcm)
        cluster_prominence = np.sum(((p + q - mean_row - mean_col) ** 4) * glcm) / np.sum(glcm) ** 2
        return cluster_prominence


    def calculate_cluster_shade(self,glcm):
        # Calculate cluster shade
        p = np.indices(glcm.shape)[0]
        q = np.indices(glcm.shape)[1]
        mean_row = np.sum(p * glcm) / np.sum(glcm)
        mean_col = np.sum(q * glcm) / np.sum(glcm)
        cluster_shade = np.sum(((p + q - mean_row - mean_col) ** 3) * glcm) / np.sum(glcm) ** 2
        return cluster_shade


    def calculate_max_probability(self,glcm):
        # Calculate max probability
        max_probability = np.max(glcm) / np.sum(glcm)
        return max_probability


    def calculate_sum_average(self,glcm):
        # Calculate sum average
        p = np.indices(glcm.shape)[0]
        q = np.indices(glcm.shape)[1]
        sum_average = np.sum((p + q) * glcm) / np.sum(glcm)
        return sum_average


    def calculate_sum_variance(self,glcm):
        # Calculate sum variance
        p = np.indices(glcm.shape)[0]
        q = np.indices(glcm.shape)[1]
        sum_average = self.calculate_sum_average(glcm)
        sum_variance = np.sum(((p + q) - sum_average) ** 2 * glcm) / np.sum(glcm)
        return sum_variance


    def calculate_sum_entropy(self,glcm):
        # Calculate sum entropy
        glcm_normalized = glcm / np.sum(glcm)
        sum_entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))
        return sum_entropy


    def calculate_difference_variance(self,glcm):
        # Calculate difference variance
        p = np.indices(glcm.shape)[0]
        q = np.indices(glcm.shape)[1]
        difference_variance = np.sum(((p - q) ** 2) * glcm) / np.sum(glcm)
        return difference_variance


    def calculate_difference_entropy(self,glcm):
        # Calculate difference entropy
        glcm_normalized = glcm / np.sum(glcm)
        difference_entropy = -np.sum(glcm_normalized * np.log2(glcm_normalized + 1e-10))
        return difference_entropy
    
    def extract_gabor_features(self, image, mask):
            features = []
            num_orientations = 8
            frequency = 0.6
            theta_values = np.arange(0, np.pi, np.pi / num_orientations)
            for theta in theta_values:
                gabor_image, _ = gabor(image, frequency=frequency, theta=theta)
                gabor_features = np.mean(gabor_image[mask == 255]), np.std(gabor_image[mask == 255])
                features.extend(gabor_features)
            return features
        
    def extract_glcm_features(self,roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Param:
        # source image
        # List of pixel pair distance offsets - here 1 in each direction
        # List of pixel pair angles in radians
        graycom = feature.graycomatrix(gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256)

        # Find the GLCM properties
        contrast = feature.graycoprops(graycom, 'contrast')
        dissimilarity = feature.graycoprops(graycom, 'dissimilarity')
        homogeneity = feature.graycoprops(graycom, 'homogeneity')
        energy = feature.graycoprops(graycom, 'energy')
        correlation = feature.graycoprops(graycom, 'correlation')
        asm = feature.graycoprops(graycom, 'ASM')

        cluster_prominence = self.calculate_cluster_prominence(graycom)
        cluster_shade = self.calculate_cluster_shade(graycom)
        max_probability = self.calculate_max_probability(graycom)
        sum_average = self.calculate_sum_average(graycom)
        sum_variance = self.calculate_sum_variance(graycom)
        sum_entropy = self.calculate_sum_entropy(graycom)
        difference_variance = self.calculate_difference_variance(graycom)
        difference_entropy = self.calculate_difference_entropy(graycom)

        # # Calculate the lacunarity feature from the contrast
        lacunarity = 1 - (contrast.var() / contrast.mean() ** 2)

        # Concatenate all the features into a single vector
        features = np.concatenate([
            contrast.ravel(),
            dissimilarity.ravel(),
            homogeneity.ravel(),
            energy.ravel(),
            correlation.ravel(),
            asm.ravel(),
            lacunarity.ravel(),
            cluster_prominence.ravel(),
            cluster_shade.ravel(),
            max_probability.ravel(),
            sum_average.ravel(),
            sum_variance.ravel(),
            sum_entropy.ravel(),
            difference_variance.ravel(),
            difference_entropy.ravel()
        ])
        # print(features)

        return features

    @staticmethod
    def extractLBPFeatures(image, mask):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp_image = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
        lbp_hist = np.histogram(lbp_image, bins=int(lbp_image.max()+1))
        return lbp_hist[0]

    def extract3DHistogram(self, image, mask):
        channels = [0, 1, 2]
        bins = [3, 3, 3]
        ranges = [0, 256, 0, 256, 0, 256]
        bgr_histogram = cv2.calcHist([image], channels, None, bins, ranges)
        features = bgr_histogram.flatten()
        return features

    def color_features(self, image, mask):
        maskb = cv2.bitwise_not(mask.astype(np.uint8)*255)
        # Ensure the mask is binary (values 0 and 255)
        maskc = cv2.threshold(maskb, 128, 255, cv2.THRESH_BINARY)[1]
        

        # Extract texture features
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp_image = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
        lbp_hist = np.histogram(lbp_image[maskc == 255], bins=np.arange(0, 60, 1))
        # Extract color features

        masked_image = cv2.bitwise_and(image, image, mask=maskb)
        mean_color = list(cv2.mean(masked_image))

        # Extract shape features
        contours, _ = cv2.findContours(maskc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, closed=True)
        area = cv2.contourArea(largest_contour)
        circularity = 4 * np.pi * area / (perimeter ** 2)

        

        gabor_features = self.extract_gabor_features(gray_image, mask)

        # Combine all features into a list
        all_features = lbp_hist[0].tolist() + mean_color + [area, perimeter, circularity] 

        # Save features to a CSV file
        output_file = 'features.csv'
        with open(output_file, 'a') as f:  # Open the file in 'append' mode to add new lines
            f.write(','.join(map(str, all_features)) + '\n')

        print(f"Features saved to {output_file}")