import numpy as np
import matplotlib.pyplot as plt
from .canny import CannyEdgeDetector
from .hough_lines import HoughLines
import cv2

class LaneDetector:

    def __init__(self, canny_min_threshold, canny_max_threshold, gaussian_kernel_size, gaussian_kernel_sigma=None, rho_bin_size=1, theta_bin_size=1, line_threshold=100, slope_threshold=0.1):

        self.canny_min_threshold = canny_min_threshold
        self.canny_max_threshold = canny_max_threshold
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_kernel_sigma = gaussian_kernel_sigma
        self.rho_bin_size = rho_bin_size
        self.theta_bin_size = theta_bin_size
        self.line_threshold = line_threshold
        self.slope_threshold = slope_threshold


    def crop_roi(self, img):

        mask = np.zeros_like(img)
        limit = int(img.shape[0]*3/5)
        mask[limit:, :] = img[limit:, :]
        return mask


    def display_lanes(self, img_rgb, line_endpoints):

        img_with_lines = np.copy(img_rgb)
        left_lane_intercepts = []
        left_lane_slopes = []
        right_lane_intercepts = []
        right_lane_slopes = []
        line_mask = np.zeros_like(img_with_lines)
        lanes_xy = [None, None]
        # line_debug = np.copy(img_rgb)


        for x1, y1, x2, y2 in line_endpoints:
            # cv2.line(line_debug, (x1, y1), (x2, y2), (0, 0, 255), 5)
            if x2 == x1:
                continue

            slope = ((img_with_lines.shape[0] - y2) - (img_with_lines.shape[0] - y1)) / (x2 - x1)
            if abs(slope) < self.slope_threshold:
                continue

            x_intercept = (slope*x1 - (img_with_lines.shape[0] - y1)) / slope
            if x_intercept > img_with_lines.shape[1] or x_intercept < 0:
                continue

            y_intercept = (img_with_lines.shape[0] - y1) - slope*x1

            if slope < 0:
                right_lane_slopes.append(slope)
                right_lane_intercepts.append(y_intercept)
            else:
                left_lane_slopes.append(slope)
                left_lane_intercepts.append(y_intercept)

        if len(right_lane_intercepts) != 0:
            right_lane_mean_intercept = np.median(right_lane_intercepts)
            right_lane_mean_slope = np.median(right_lane_slopes)
            right_y1 = img_with_lines.shape[0]
            right_y2 = int(img_with_lines.shape[0]*3/5)
            right_x1 = int((-1*right_lane_mean_intercept) / right_lane_mean_slope)
            right_x2 = int((img_with_lines.shape[0]*2/5 - right_lane_mean_intercept) / right_lane_mean_slope)
            cv2.line(line_mask, (right_x1, right_y1), (right_x2, right_y2), (0, 0, 255), 5)
            lanes_xy[1] = (right_x1, right_y1, right_x2, right_y2)

        if len(left_lane_intercepts) != 0:
            left_lane_mean_intercept = np.median(left_lane_intercepts)
            left_lane_mean_slope = np.median(left_lane_slopes)
            left_y1 = img_with_lines.shape[0]
            left_y2 = int(img_with_lines.shape[0]*3/5)
            left_x1 = int((-1*left_lane_mean_intercept) / left_lane_mean_slope)
            left_x2 = int((img_with_lines.shape[0]*2/5 - left_lane_mean_intercept) / left_lane_mean_slope)
            cv2.line(line_mask, (left_x1, left_y1), (left_x2, left_y2), (0, 0, 255), 5)
            lanes_xy[0] = (left_x1, left_y1, left_x2, left_y2)

        img_with_lines = cv2.addWeighted(img_with_lines, 0.6, line_mask, 0.4, 0)

        # plt.imshow(line_debug)
        # plt.show()

        return img_with_lines, lanes_xy


    def detect_lanes(self, img_rgb):

        canny_edge_detector = CannyEdgeDetector(self.canny_min_threshold, self.canny_max_threshold, self.gaussian_kernel_size, self.gaussian_kernel_sigma)
        img_canny = canny_edge_detector.apply_canny(img_rgb)

        # plt.imshow(img_canny, cmap="gray")
        # plt.show()

        img_canny_cropped = self.crop_roi(img_canny)
        
        # plt.imshow(img_canny_cropped, cmap="gray")
        # plt.show()

        hough_line_detector = HoughLines(self.rho_bin_size, self.theta_bin_size, self.line_threshold)
        line_endpoints = hough_line_detector.apply_hough_lines(img_canny_cropped)

        img_with_lanes, lanes_xy = self.display_lanes(img_rgb, line_endpoints)

        return img_with_lanes, lanes_xy


