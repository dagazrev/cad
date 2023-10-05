import cv2
import numpy as np

class Preprocessing:
    def __init__(self):
        pass

    def preprocessApproach1(self, image):
        return self.hairRemoval(image)
        pass

    def preprocessApproach2(self, image):
        pass

    def hairRemoval(self, img):
        # Switch to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create 16 linear structuring elements (SEs)
        linearSEs = [cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))]

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
        inpainted_image = cv2.inpaint(img, sum_black_hats, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

        return inpainted_image
        pass