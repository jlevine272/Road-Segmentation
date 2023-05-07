import numpy as np
import matplotlib.pyplot as plt

class HoughLines:

    def __init__(self, rho_bin_size=1, theta_bin_size=1, line_threshold=100):
        
        self.rho_bin_size = rho_bin_size
        self.theta_bin_size = theta_bin_size
        self.line_threshold = line_threshold


    def collect_lines(self, img):
    
        hypotenuse_length = np.ceil(np.sqrt(img.shape[0]**2 + img.shape[1]**2))
        row_indices, col_indices = np.where(img != 0)

        num_rhos = (hypotenuse_length + 1 + hypotenuse_length) // self.rho_bin_size
        rho_values_list = np.empty(int(num_rhos), dtype=np.int32)
        for i in range(int(num_rhos)):
            rho_values_list[i] = -1*hypotenuse_length + i * self.rho_bin_size
        
        num_thetas = 180 // self.theta_bin_size
        theta_values_list = np.empty(num_thetas, dtype=np.int32)
        for i in range(num_thetas):
            theta_values_list[i] = -90 + i * self.theta_bin_size

        theta_values_list = np.deg2rad(theta_values_list)
        lines_collection = np.zeros((len(rho_values_list), len(theta_values_list)), dtype=np.uint64)

        for row, col in zip(row_indices, col_indices):
            for theta_idx in range(len(theta_values_list)):
                distance_from_origin = int((col * np.cos(theta_values_list[theta_idx]) + row * np.sin(theta_values_list[theta_idx])) + hypotenuse_length)
                lines_collection[distance_from_origin, theta_idx] += 1

        return lines_collection, rho_values_list, theta_values_list


    def compute_endpoints(self, lines_collection, rho_values_list, theta_values_list):
    
        line_indices = np.where(lines_collection >= self.line_threshold)

        line_endpoints = []
        
        for rho_idx, theta_idx in zip(line_indices[0], line_indices[1]):

            rho = rho_values_list[rho_idx]
            theta = theta_values_list[theta_idx]
            
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            x0 = rho * cos_theta
            y0 = rho * sin_theta
            x1 = int(x0 + 1000 * (-sin_theta))
            y1 = int(y0 + 1000 * (cos_theta))
            x2 = int(x0 - 1000 * (-sin_theta))
            y2 = int(y0 - 1000 * (cos_theta))

            line_endpoints.append((x1, y1, x2, y2))

        return line_endpoints


    def apply_hough_lines(self, img):

        lines_collection, rho_values_list, theta_values_list = self.collect_lines(img)
        line_endpoints = self.compute_endpoints(lines_collection, rho_values_list, theta_values_list)

        return line_endpoints

    