import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from scipy.signal import convolve2d


class CannyEdgeDetector:


    def __init__(self, min_threshold, max_threshold, gaussian_kernel_size, gaussian_kernel_sigma=None):
        
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_kernel_sigma = gaussian_kernel_sigma


    def get_gaussian_kernel(self):
    
        half_size = self.gaussian_kernel_size // 2
        
        if self.gaussian_kernel_sigma == None:
            self.gaussian_kernel_sigma = 0.3*((self.gaussian_kernel_size-1)*0.5 - 1) + 0.8
        
        x, y = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]
        gaussian_kernel = np.exp(-((x**2 + y**2) / (2.0*self.gaussian_kernel_sigma**2))) / (2 * np.pi * self.gaussian_kernel_sigma**2)
        
        return gaussian_kernel


    def apply_gaussian(self, img):

        gaussian_kernel = self.get_gaussian_kernel()
        img_convolved = convolve2d(img, gaussian_kernel, mode='same')

        return img_convolved 


    def get_sobel_kernels(self):

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        return sobel_x, sobel_y

    
    def get_sobel_gradients(self, img):

        sobel_x, sobel_y = self.get_sobel_kernels()

        gradient_img_x = convolve2d(img, sobel_x, mode="same")
        gradient_img_y = convolve2d(img, sobel_y, mode="same")

        gradient_img = np.sqrt(gradient_img_x**2 + gradient_img_y**2)
        gradient_img = (gradient_img / np.amax(gradient_img)) * 255.0

        gradient_angles = np.arctan2(gradient_img_y, gradient_img_x)

        return gradient_img, gradient_angles

    
    def non_max_suppression(self, img, gradient_angles):

        post_nms = np.zeros_like(img, dtype=np.int32)
        gradient_angles_in_degrees = np.degrees(gradient_angles) % 180

        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                if (0 <= gradient_angles_in_degrees[i, j] < 22.5) or (157.5 <= gradient_angles_in_degrees[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (22.5 <= gradient_angles_in_degrees[i, j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif (67.5 <= gradient_angles_in_degrees[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (112.5 <= gradient_angles_in_degrees[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    post_nms[i, j] = img[i, j]

        return post_nms

    def apply_thresholding(self, img):

        self.max_threshold = np.max(img) * self.max_threshold
        self.min_threshold = self.max_threshold * self.min_threshold

        thresholded_image = np.zeros_like(img, dtype=np.int32)
        thresholded_image[img >= self.max_threshold] = 255
        thresholded_image[(img < self.max_threshold) & (img >= self.min_threshold)] = -999

        dx = [-1, 0, 1]
        dy = [-1, 0, 1]

        for i in range(thresholded_image.shape[0]):
            for j in range(thresholded_image.shape[1]):
                if thresholded_image[i,j] == -999:
                    neighbor_is_strong = False
                    for x in dx:
                        for y in dy:
                            if thresholded_image[i+x, j+y] == 255:
                                neighbor_is_strong = True
                    if neighbor_is_strong:
                        thresholded_image[i,j] = 255
                    else:
                        thresholded_image[i,j] = 0

        return thresholded_image

    
    def apply_canny(self, img_rgb):

        img_gray = color.rgb2gray(img_rgb)
        img_convolved = self.apply_gaussian(img_gray)
        img_sobel, gradient_angles = self.get_sobel_gradients(img_convolved)
        img_post_nms = self.non_max_suppression(img_sobel, gradient_angles)
        img_thresholded = self.apply_thresholding(img_post_nms)

        return img_thresholded
