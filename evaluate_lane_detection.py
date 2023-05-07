from Lane_Detection.lane_detection import LaneDetector
from llamas_evaluation import evaluate
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
import os
import json

images_path = "/data3/sb56/CP_proj/color_images/color_images/valid/images-2014-12-22-14-19-07_mapping_280S_3rd_lane/"
ld = LaneDetector(0.05, 0.09, 7, line_threshold=100)

def get_x_values(lines_xy):

    l0_x_values = np.array([])
    if lines_xy[0] is not None:
        l0_y_values = np.arange(301, 717 + 1)
        l0_x_values = ((l0_y_values - 717) * (lines_xy[0][2] - lines_xy[0][0]) / (lines_xy[0][3] - 717) + lines_xy[0][0]).astype(int)

    r0_x_values = np.array([])
    if lines_xy[1] is not None:
        r0_y_values = np.arange(301, 717 + 1)
        r0_x_values = ((r0_y_values - 717) * (lines_xy[1][2] - lines_xy[1][0]) / (lines_xy[1][3] - 717) + lines_xy[1][0]).astype(int)

    return l0_x_values, r0_x_values


if __name__ == "__main__":
    
    file_counter = 0
    data = {}

    for file_name in os.listdir(images_path):
        
        file_path = os.path.join(images_path, file_name)

        img_rgb = mimg.imread(file_path)
        lane_img, lines_xy = ld.detect_lanes(img_rgb)
        l0, r0 = get_x_values(lines_xy)

        key = '/'.join(file_path.split("/")[-2:])[:-15] + ".json"
        x_values = {
            "l0": l0.tolist(),
            "r0": r0.tolist()
        }
        data[key] = x_values
        file_counter += 1

        if file_counter % 1 == 0:
            print(file_counter)

    json_data = json.dumps(data, indent=4)

    with open('output.json', 'w') as file:
        file.write(json_data)

    evaluate.evaluate("output.json", "valid")
    

