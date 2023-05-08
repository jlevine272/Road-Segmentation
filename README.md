# Requirements
Before running this code, please install all of the packages in `requirements.txt`.

# Structure
The main file of this project is `main.ipynb`, which contains the pipeline to run semantic segmentation and lane detection on a directory of images. To get output images, run this notebook, setting `im_dir` to the appropriate directory. Then update the info variable to reflect the statistics of the dataset you are using.

If you would like to swap the semantic segmentation model, you must edit the function semantic_segmentation in `segmentation_proj.py`, such that the parameters and outputs are of the same type.

## Semantic Segmentation
This notebook uses code from Fisher Yu to run segmentation. The segmentation architecture is in `segment.py`, and the `semantic_segmentation` function in `segment_proj.py` drives the segmentation. 

## Lane Detection
Lane detection is implemented using Canny Edge Detection and Hough Transform, which are implemented in `Lane_Detection/canny.py` and `Lane_Detection/hough_lines.py`, respectively. The two components are combined in `lane_detection.py`.

Lane detection was evaluated using `evaluate_lane_detection.py`.

## Debugging
While debugging our code, we used `lane_detect.ipynb`, which gives outputs for the lane detection dataset.
