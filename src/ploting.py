from matplotlib import pyplot as plt
import matplotlib as mpl
import cv2
import numpy as np

class Ploting:
    @staticmethod
    def showImage(lesionImage):
        plt.imshow(lesionImage, cmap="gray")
        plt.axis("off")
        plt.show()

    @staticmethod
    def contours(lesionImage, contours):
        fig, ax = plt.subplots()
        ax.imshow(lesionImage, cmap="gray")
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        plt.show()

    @staticmethod
    def comparison(lesionImage, GT, pred):
        
        # for coloring
        GT_inv = (np.logical_not(GT)).astype("uint8")
        pred_inv = (np.logical_not(pred)).astype("uint8")
        GT = GT.astype("uint8")
        pred = pred.astype("uint8")

        tp = cv2.bitwise_and(GT,pred)
        fp = cv2.bitwise_and(GT_inv,pred)
        fn = cv2.bitwise_and(GT,pred_inv)

        cmap1 = mpl.colors.ListedColormap(['none', 'green'])
        cmap2 = mpl.colors.ListedColormap(['none', 'red'])
        cmap3 = mpl.colors.ListedColormap(['none', 'blue'])

        plt.imshow(lesionImage)
        opacity = 0.5
        plt.imshow(tp,  cmap=cmap1, alpha=opacity*(tp>0), interpolation="none")
        plt.imshow(fp,  cmap=cmap2, alpha=opacity*(fp>0), interpolation="none")
        plt.imshow(fn,  cmap=cmap3, alpha=opacity*(fn>0), interpolation="none")

        patch1 = plt.Rectangle((0, 0), 1, 1, color=cmap1(1.0))
        patch2 = plt.Rectangle((0, 0), 1, 1, color=cmap2(1.0))
        patch3 = plt.Rectangle((0, 0), 1, 1, color=cmap3(1.0))

        # add the legend to the plot
        plt.legend([patch1, patch2, patch3], ["True Positive", "False Positive","False Negative"], loc='upper right')
        plt.axis("off")
        plt.show()
    
    @staticmethod
    def inputOutput(input, output):
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(input)
        axes[0].axis('off')

        axes[1].imshow(output, cmap="gray")
        axes[1].axis('off')

        plt.subplots_adjust(wspace=0.05)
        plt.show()
    