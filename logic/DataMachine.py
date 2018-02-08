from skimage import io
from skimage import feature
import cv2
import matplotlib.pyplot as plt

class DataMachine:
    def __init__(self, input_image):
        self.input_image = input_image

    def prepare_data(self):
        # im = io.imread(self.input_image)
        gray_image = self.__convert_gray_scale()
        cv2.imshow('gray_image', gray_image)
        cv2.waitKey(0)  # Waits forever for user to press any key
        cv2.destroyAllWindows()
        edges = feature.canny(gray_image, sigma=3.)
        io.imshow(edges)
        io.show()

        cv2.imwrite('../data/output_image.png', gray_image)
        plt.imsave('../data/input_image.png', edges, cmap=plt.cm.gray)


    def __convert_gray_scale(self):
        image = cv2.imread(self.input_image)
        cv2.imshow('color_image', image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

