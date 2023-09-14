import numpy as np
from tqdm import tqdm
from registration.align_utils import load, preprocess, bbox, find_lines, process_lines, svd_fit, get_affine 

def vol2affine(moving: np.ndarray = "data/brain_25.nii.gz", 
               template: np.ndarray = "data/template/average_template_25.nii.gz",
               pivot = (0, 0, 0)):
    """
    Given two volumes, find the affine matrix that aligns the plane
    corresponding to the longitudinal fissure for the moving dataset
    to the template dataset. Pivot is the point that the volume is 
    rotated around. By default, it rotates about the origin.
    """
    # If the parameter is a string, load it to memory.
    if type(moving) == str:
        if '.npy' in moving:
            moving = np.load(moving)
        elif '.nii.gz' in moving:
            moving = load(moving)
    if type(template) == str:
        if '.npy' in template:
            template = np.load(template)
        elif '.nii.gz' in template:
            template = load(template)
            
    # Below listed are parameters used in tweaking results. 
    # Default values work best for the provided data "brain_25.nii.gz"
    # and template "average_template_25.nii.gz"
    # Variables are provided for both moving and template operations
    # but they can be mostly kept the same values as each other,
    # except for moving_range and template_range to remove outliers.
    
    ###########################################
    
    debug = False  # Set to True to print and show intermediary steps for debugging.
    # Maximum pixel value to cap the data at before normalization. 
    max_val = 400; max_val_template = 400
    # Once the bounding box has been found, only lines within a certain region
    # of the bounding box will be considered valid. This area starts at the midpoint
    # of the bounding box and extends both left and right on the x-axis by a total of
    # bbox_width * ratio * 2 pixels
    ratio = 0.05; ratio_template = 0.05
    # Range of indices to look for line segments [inclusive, exclusive). 
    # Indices outside of this range are considered outliers and discarded.
    moving_range = (41, 202); template_range = (0, template.shape[0])
    
    # Line detection parameters
    min_thresh = 150; min_thresh_template = 150
    max_thresh = 250; max_thresh_template = 250
    # Size of Gaussian blur kernel operation run before Canny operation.
    # Skips Gaussian blur operation if 0
    blur_kernel_size = 0; blur_kernel_size_template = 0
    # Used for cv2.HoughLinesP
    rho = 1; rho_template = 1
    # The resolution of detected lines (default pi/180 = 1 degree)
    theta = np.pi / 180; theta_template = np.pi / 180
    # The minimum number of intersections to detect a line in HoughP
    line_thresh = 15; line_thresh_template = 15
    # The minimum number of points that can form a line. 
    # Lines with less than this number of points are disregarded
    min_line_length = 30; min_line_length_template = 30
    # The maximum gap between two points to be considered in the same line
    max_line_gap = 20; max_line_gap_template = 20
    # Only line segments that are oriented between these angles (degrees) are accepted. 
    min_angle = 80; min_angle_template = 80
    max_angle = 100; max_angle_template = 100
    
    ###########################################
    
    # Preprocess data
    moving_copy = np.copy(moving) 
    moving_copy = preprocess(moving_copy, max_val=max_val)
    template_copy = np.copy(template)
    template_copy = preprocess(template_copy, max_val=max_val)
    
    # Store detected line points here
    moving_points = []
    template_points = []
    
    # Detect lines from longitudinal fissure
    for i in tqdm(range(moving_range[0], moving_range[1])):
        curr_img = moving_copy[i, :, :]
        left, _, right = bbox(curr_img, ratio=ratio, debug=debug)
        lines = find_lines(curr_img, 
                           min_thresh=min_thresh, max_thresh=max_thresh, 
                           blur_kernel_size=blur_kernel_size,
                           rho=rho, theta=theta, line_thresh=line_thresh, 
                           min_line_length=min_line_length, max_line_gap=max_line_gap, debug=debug)
        lines, avg_line = process_lines(lines, left, right, 
                                        min_angle=min_angle, max_angle=max_angle, 
                                        debug=debug)
        if avg_line:  # If a line is detected, add it.
            avg_line = np.squeeze(avg_line)
            x1, y1, x2, y2 = avg_line
            # Set up a matrix for the plane equation: ax + by + cz = 0
            moving_points.append([i, y1, x1])
            moving_points.append([i, y2, x2])
    moving_points = np.array(moving_points)
        
    # Repeat for template
    for i in tqdm(range(template_range[0], template_range[1])):
        curr_img = template_copy[i, :, :]
        left, _, right = bbox(curr_img, ratio=ratio, debug=debug)
        lines = find_lines(curr_img, 
                           min_thresh=min_thresh_template, max_thresh=max_thresh_template, 
                           blur_kernel_size=blur_kernel_size_template,
                           rho=rho_template, theta=theta_template, 
                           line_thresh=line_thresh_template, 
                           min_line_length=min_line_length_template, 
                           max_line_gap=max_line_gap_template, debug=debug)
        lines, avg_line = process_lines(lines, left, right, 
                                        min_angle=min_angle_template, 
                                        max_angle=max_angle_template, debug=debug)
        if avg_line:  # If a line is detected, add it.
            avg_line = np.squeeze(avg_line)
            x1, y1, x2, y2 = avg_line
            # Set up a matrix for the plane equation: ax + by + cz = 0
            template_points.append([i, y1, x1])
            template_points.append([i, y2, x2])
    template_points = np.array(template_points)
    
    # Use SVD to fit a plane ax + by + cz = d to the retrieved points
    a, b, c, _ = svd_fit(moving_points, debug=debug)
    ta, tb, tc, _ = svd_fit(template_points, debug=debug)
    coef = np.array([a, b, c])
    template_coef = np.array([ta, tb, tc])
    
    # Get affine matrix to fit the moving normal vector to the template normal vector
    affine = get_affine(coef, template_coef, pivot=pivot)
    return affine, coef, template_coef