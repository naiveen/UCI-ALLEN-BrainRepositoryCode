import math
import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def load(filepath="brain_25.nii.gz"):
    """
    Loads brain nii.gz file from a given filepath.
    """
    return nib.load(filepath).get_fdata()


def preprocess(data, max_val=400):
    """
    Preprocesses volume data. Clips maximum value at max_val and then normalizes volume
    between 0-255.
    """
    data[data > max_val] = max_val
    data = cv2.normalize(src=data, dst=None, alpha=0, beta=255, 
                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return data
    
    
def info(data):
    """
    Show information about the input data.
    """
    print("Shape:", data.shape)
    print("Min:", np.min(data))
    print("Max:", np.max(data))
    print("Mean:", np.mean(data))
    print("Median:", np.median(data))
    
    
def show(img, title=""):
    """
    Quickly displays the input image.
    """
    plt.figure(figsize=(6,6))
    plt.axis(False)
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.show()
    
    
def bbox(img, ratio=0.15, debug=False):
    """
    Computes a bounding box around a given image.
    The parameter ratio is used to compute a space of size 2 * ratio * bounding box width
    centered at the bounding box's midpoint. Lines that fall inside this space are valid
    and kept.
    """
    # Otsu threshold the image.
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #kernel = np.ones((6, 6), np.uint8)
    #thresh = cv2.erode(thresh, kernel)
    if debug:
        show(thresh, "Thresholded")

    # Find contours of thresholded image, obtain bounding box, extract ROI
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if debug:
        print("Num of bounding boxes:", len(cnts))
    
    # Left and right boundaries for x-values. 
    # Line segments that fall within these left/right boundaries are valid.
    left = -1; right = -1; midpoint = -1
    largest_area = -1
    boundary = img
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Retrieve the largest contour and use that as our bounding box.
        # Assume the largest bounding box is what we want to work with.
        if w * h > largest_area:
            largest_area = w * h
            midpoint = x + (w/2)
            left = midpoint - ratio * w
            right = midpoint + ratio * w
            img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            boundary = cv2.rectangle(img_copy, (x, y), (x + w, y + h), (36, 255, 12), 2)
    if debug and cnts:
        print("x, y, w, h", x, y, w, h)
        print("midpoint:", midpoint)
        print("left:", left)
        print("right:", right)
        show(boundary, "Bounding box")
    return left, midpoint, right
    
    
def bbox_vert(img, up_ratio=0.05, bottom_ratio=0.35, debug=False):
    """
    Computes a bounding box around a given image. Vertical version of bbox function.
    The parameter ratios are used to compute a space of size 2 * ratio * bounding box height
    centered at the bounding box's midpoint. Lines that fall inside this space are valid
    and kept.
    """
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Dilate the output so very close pieces are connected.
    kernel = np.ones((6, 6), np.uint8)
    thresh = cv2.dilate(thresh, kernel)
    if debug:
        show(thresh, "Thresholded + Dilation")

    # Find contours, obtain bounding box, extract ROI
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if debug:
        print("Num of bounding boxes:", len(cnts))
    
    # Up and bottom boundaries for x-values. 
    # Line segments that fall within these up/bottom boundaries are valid.
    up = -1; bottom = -1; midpoint = -1
    largest_area = -1
    boundary = img
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Retrieve the largest contour and use that as our bounding box.
        # Assume the largest bounding box is what we want to work with.
        if w * h > largest_area:
            largest_area = w * h
            midpoint = y + (h/2)
            up = midpoint - up_ratio * h
            bottom = midpoint + bottom_ratio * h
            img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            boundary = cv2.rectangle(img_copy, (x, y), (x + w, y + h), (36, 255, 12), 2)
    if debug and cnts:
        print("x, y, w, h", x, y, w, h)
        print("midpoint:", midpoint)
        print("up:", up)
        print("bottom:", bottom)
        show(boundary, "Bounding box")
    return up, midpoint, bottom

    
def find_lines(img, min_thresh=150, max_thresh=250, blur_kernel_size=5,
               rho=1, theta=np.pi/180, line_thresh=15, min_line_length=30, max_line_gap=20,
               debug=False):
    """
    Given an normalized input image, run line detection to retrieve salient lines from the
    image. 
    """
    # Find canny edges
    img_copy = img.copy()
    # Blur the kernel only if size > 0. Blur may be counterproductive for horizontal line
    # detection casees. 
    if blur_kernel_size:
        img_copy = cv2.GaussianBlur(img_copy, (blur_kernel_size, blur_kernel_size), 0)
    edge_img = cv2.Canny(img_copy, min_thresh, max_thresh, None, 3)
    if debug:
        show(edge_img, "Edge")
    
    # Find Hough lines and return a list of lines.
    lines = cv2.HoughLinesP(edge_img, rho, theta, line_thresh, np.array([]), 
                            min_line_length, max_line_gap)
    return lines
    

def process_lines(lines, left, right, min_angle=80, max_angle=100, debug=False):
    """
    Once we get our lines from find_lines, process them and keep the lines
    that are of a desired angle that fall within valid boundaries.
    """
    valid_lines = []
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            # Here l contains x1,y1,x2,y2  of your line
            # so you can compute the orientation of the line 
            x1 = l[0]; y1 = l[1]
            x2 = l[2]; y2 = l[3]
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            p3 = np.subtract( p2, p1 ) # Translate p2 by p1
            # Compute line segment angle properties.
            angle_radians = math.atan2(p3[1], p3[0])
            angle_degree = angle_radians * 180 / math.pi
            if debug:
                print("l:", lines[i])
                print("degree", angle_degree)
            # If the current line segment is valid, add it
            in_bounds = x1 >= left and x1 <= right and x2 >= left and x2 <= right
            if abs(angle_degree) > min_angle and abs(angle_degree) < max_angle and in_bounds:
                curr_line = lines[i]
                # Switch the coordinates if the angle computation is negative to 
                # avoid issues with averaging lines.
                if angle_degree < 0:
                    curr_line = [[x2, y2, x1, y1]]
                valid_lines.append(curr_line)
    # Compute the average line.
    if debug:
        print("valid lines:")
        print(valid_lines)
    averaged_line = valid_lines
    # If multiple lines have been retrieved from the computation, 
    # find the average line of the set.
    if len(valid_lines) > 1:
        v = np.array(valid_lines)
        averaged_line = [[list(np.mean(np.squeeze(v), axis=0).astype(np.uint16))]]
    if debug:
        print("avg line:")
        print(averaged_line)
    return valid_lines, averaged_line


def process_lines_vert(lines, up, bottom, min_angle=-30, max_angle=30, debug=False):
    """
    Vertical version of process_lines function.
    """
    valid_lines = []
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            # Here l contains x1,y1,x2,y2  of your line
            # so you can compute the orientation of the line 
            x1 = l[0]; y1 = l[1]
            x2 = l[2]; y2 = l[3]
            p1 = np.array([x1, y1])
            p2 = np.array([x2, y2])
            p3 = np.subtract( p2, p1 ) # Translate p2 by p1
            # Compute line segment angle properties.
            angle_radians = math.atan2(p3[1], p3[0])
            angle_degree = angle_radians * 180 / math.pi
            if debug:
                print("l:", lines[i])
                print("degree", angle_degree)
            # If the current line segment is valid, add it
            in_bounds = y1 >= up and y1 <= bottom and y2 >= up and y2 <= bottom
            if abs(angle_degree) > min_angle and abs(angle_degree) < max_angle and in_bounds:
                curr_line = lines[i]
                # Switch the coordinates if the angle computation is negative to 
                # avoid issues with averaging lines.
                if angle_degree < 0:
                    curr_line = [[x2, y2, x1, y1]]
                valid_lines.append(curr_line)
    # Compute the average line
    if debug:
        print("valid lines:")
        print(valid_lines)
    averaged_line = valid_lines
    # If multiple lines have been retrieved from the computation, 
    # find the average line of the set.
    if len(valid_lines) > 1:
        v = np.array(valid_lines)
        averaged_line = [[list(np.mean(np.squeeze(v), axis=0).astype(np.uint16))]]
    if debug:
        print("avg line:")
        print(averaged_line)
    return valid_lines, averaged_line


def create_line_img(img, lines):
    """
    Overlay a list of lines over an input image and return the output image.
    """
    img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    line_image = np.copy(img_copy) * 0  # creating a blank to draw lines on
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),5)
    # Draw the lines on the image
    return cv2.addWeighted(img_copy, 0.8, line_image, 1, 0)


def svd_fit(filepath, debug=False):
    """
    Given a filepath or set of points, use SVD to fit a plane to them.
    """
    if type(filepath) == str:  # If string is passed, load the .npy
        points = np.load(filepath)
    else:  # Otherwise assume it is a NumPy array
        points = filepath
    num_points = points.shape[0]
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Perform Singular Value Decomposition
    centroid = np.mean(points, axis=0)
    if debug:
        print('centroid:', centroid)
    centered_points = points - centroid
    covariance_matrix = centered_points.T @ centered_points
    _, _, vh = np.linalg.svd(covariance_matrix)

    # The normal vector of the plane is the smallest singular vector (last column of vh)
    normal = vh[-1]

    # Normalize the normal vector
    normal /= np.linalg.norm(normal)
    
    # Now you have the normal vector and a point on the plane
    # The plane's equation can be represented as ax + by + cz = d
    a, b, c = normal
    d = np.dot(normal, centroid)

    # The plane equation coefficients are now a, b, c, and d
    if debug:
        print(f"Plane equation: {a}x + {b}y + {c}z = {d}")
        print(f"{a}, {b}, {c}, {d}")
        print("d * normal:", np.array([a, b, c]) * d)
    
    # Create a meshgrid of x, y, z values
    x_vals = np.linspace(min(x), max(x), 50)
    y_vals = np.linspace(min(y), max(y), 50)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_grid = (-a * x_grid - b * y_grid + d) / c

    # Create a 3D plot
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plot the points
        ax.scatter(x, y, z, c='r', marker='o', label='Points')
        # Plot the fitted plane
        plane_surface = ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, color='b')
        custom_legend_entry = plt.Line2D([0], [0], linestyle="none", c='b', 
                                         marker='o', markersize=10, markerfacecolor='b', 
                                         label='Fitted Plane')
        ax.legend(handles=[custom_legend_entry])  # Explicitly create legend with custom entry
        # Set labels for the axes
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    return np.array([a, b, c, d])


def align_rotation(v, v_target, debug=False):
    """
    Outputs a rotation matrix to align v to v_target.
    """
    v = v / np.linalg.norm(v)
    v_target = v_target / np.linalg.norm(v_target)

    # Calculate the rotation axis and angle needed to align the 
    # source normal vector n with the target normal vector n_target. 
    # You can use the cross product to find the rotation axis and 
    # the dot product to find the angle between the vectors.
    rotation_axis = np.cross(v, v_target)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot(v, v_target))

    # Create rotation transformation matrix
    rotation = R.from_rotvec(rotation_angle * rotation_axis)
    rot_matrix = rotation.as_matrix()
    
    
    
    if debug:
        print('input:', v)
        print('v_target_norm:', v_target)
        print('rot_axis:', rotation_axis)
        print('rot_angle:', rotation_angle * 180/np.pi)
        print('rot_matrix:')
        print(rot_matrix)
        
    return rot_matrix
    #rot_matrix = np.identity(3)
    # Apply shear transformations to match shear factors
    # Calculate the shear factors needed to align the source plane with the target plane. 
    # You can determine the shear factors by comparing the components of the source and 
    # target normal vectors that are not aligned with the plane.
    #shear_factors = (n_target - np.dot(n_target, n) * n) / np.dot(n, n)

    #print('shear_factors:', shear_factors)
    #shear_matrix = np.identity(3) + shear_factors[:, np.newaxis] * n
    #shear_matrix = np.identity(3)

    # Apply shear matrix and rotation matrix to your plane's normal vector
    #transformed_normal = np.dot(rot_matrix, np.dot(shear_matrix, n))

    # Now, 'transformed_normal' should be aligned with 'n_target'

    #print("transformed:", transformed_normal)
    #print("transformed norm:", transformed_normal / np.linalg.norm(transformed_normal))
    
    
def get_affine(v, v_target, pivot=(0, 0, 0)):
    """
    Get affine matrix from two normal vectors. 
    Constructs a rotation around a pivot point (x, y, z).
    """
    # Make a 4x4 rotation matrix.
    rot_matrix = align_rotation(v, v_target)
    translation = np.array([[0, 0, 0]]).T
    homogenous = np.array([[0, 0, 0, 1]])
    rot_matrix = np.concatenate([rot_matrix, translation], axis=1)
    rot_matrix = np.concatenate([rot_matrix, homogenous], axis=0)
    
    pivot = np.array(pivot)
    # Create a translation matrix to move the pivot point back to the origin
    translation_matrix_to_origin = np.identity(4)
    translation_matrix_to_origin[:3, 3] = -pivot
    
    # Create a translation matrix to move from the origin to the pivot point
    translation_matrix_from_origin = np.identity(4)
    translation_matrix_from_origin[:3, 3] = pivot
    
    # Combine the matrices to form the full affine matrix
    affine_matrix = translation_matrix_to_origin @ rot_matrix @ translation_matrix_from_origin
    return affine_matrix

