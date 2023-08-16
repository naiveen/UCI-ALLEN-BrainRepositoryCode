import scipy.io
import numpy as np
import subprocess
import argparse 
import os
import nibabel as nib
from collections import Counter
import pandas as pd
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove
from cell_counting import *
from cell_detection import *
from reconstruction import *

def get_input_cell_locations(input_mat):
    mat = scipy.io.loadmat(input_mat)
    location_arr  = mat['location'][0]
    prediction_arr = mat['predictions'][0]
    cell_locations = []
    for i in range(location_arr.shape[0]):
        location = location_arr[i]
        prediction = prediction_arr[i]
        location = np.squeeze(location)
        prediction = np.squeeze(prediction)
        if(len(location) ==0):
            continue
        true_locations = location.T[prediction!=2]
        print(true_locations.shape)
        true_locations = np.hstack((true_locations, np.zeros((true_locations.shape[0],1))+i))
        cell_locations.append(true_locations)
    cell_locations = np.vstack(cell_locations) 
    return cell_locations


def replaceBSplineOrder(file_path):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                if "FinalBSplineInterpolationOrder" in line:
                    new_file.write("(FinalBSplineInterpolationOrder 0)\n")
                else:
                    new_file.write(line)
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)

def parsePhysicalPointsFromOutputFile(transformixOutFile):
    """
    Parses Transformix Output file and stores the physical point location

    Parameters
    ----------
    transformixOutFile  : Path to transformix output file
    """
    outputPoints=[]

    with open(transformixOutFile, "r") as f:
        for line in f:
            line = line.split(";")
            outputPoint  = [float(x) for x in line[4].split()[3:6]]
            outputPoints.append(outputPoint)
    outputPoints= np.asarray(outputPoints)
    return outputPoints

def convertPhysicalPointsToIndex(outputPoints, annDataImage):
    outputIndices=[]
    for point in outputPoints:
        index = annDataImage.TransformPhysicalPointToIndex(point)
        outputIndices.append(index)
    return outputIndices

def countCellsInRegions(outputIndices, annDataImage):
    """
    Parameters:
    -----------
    outputIndices : list of tuples, each tuple is a index location
    annDataImage : sitkImage
    """
    region_ids =[]
    points =[]
    exterior_points =0
    for i,point in enumerate(outputIndices):
        try:
            point = np.asarray(point).astype(np.uint16)
            if(np.sum(point<0)!=0):
                continue

            sx, sy, sz = annDataImage.GetSize()

            if(point[1] >= sy or point[2] >= sz or point[0]>= sx):
                exterior_points = exterior_points+1
                continue

            if annDataImage.GetPixel(int(point[0]), int(point[1]), int(point[2]))>=1:
                region_ids.append(int(annDataImage.GetPixel(int(point[0]), int(point[1]), int(point[2]))))
                points.append(i)
        except Exception as e:
            print(e)
            print(i, point)
    cell_counts = Counter(region_ids)
    print("Exterior Points :{}".format(exterior_points))
    return cell_counts, np.asarray(points)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default="", help = "Path to image directory containing stitiched sections")
    parser.add_argument('--channel', type=int, default=0, help = "Channel to use for registration and cell counting")
    parser.add_argument('--cell_detection', action='store_true', help ="Use this argument to do cell counting")
    parser.add_argument('--threshold', type=int, default=10, help ="Threshold for cell counting")

    parser.add_argument('--fixed_image', type=str, default = "") #.\reference\average_template_25.nii
    parser.add_argument('--moving_image', type=str, default ="") #output\brain25\result.1.nii
    parser.add_argument('--output_dir', type=str, default = "")
    parser.add_argument('--input_mat', type=str, default = None)

    parser.add_argument('--flag', action='store_true',  help ="dummy") 

    args = parser.parse_args()
    elastix_dir = "../elastix/"
    template_dir = "CCF_DATA/"
    annotationImagePath = r"./CCF_DATA/annotation_25m.nii"
    ATLAS_PATH = r"./CCF_DATA/1_adult_mouse_brain_graph_mapping.csv"

    img_dir  = args.img_dir
    output_dir = args.output_dir
    input_mat = args.input_mat
    channel = args.channel
    threshold  = args.threshold

    if img_dir == "":
        moving_image = args.moving_image
        fixed_image = args.fixed_image
        assert moving_image!="", f"Either provide Image directory or moving_image file path"
    else:
        nii_dir  = os.path.join(img_dir, "nii")
        moving_image = os.path.join(nii_dir,"brain_25.nii.gz")
        fixed_image = os.path.join(template_dir, "average_template_25m.nii")

    if args.output_dir=="":
        output_dir = os.path.join(img_dir, "registration")

    if not os.path.isfile(moving_image):
        print("Creating Nii Images.")
        createNiiImages(img_dir, nii_dir, channel)

    input_points_file = os.path.join(output_dir, "inputpoints.txt")
    output_points_file = os.path.join(output_dir, "output_points.txt")
    cell_count_file = os.path.join(output_dir, "cell_count.csv")
    
    
    registration_cmd = [elastix_dir+"elastix","-m",fixed_image, "-f", moving_image, "-out", output_dir, "-p" , "001_parameters_Rigid.txt", "-p", "002_parameters_BSpline.txt"]
    transformix_cmd  = [elastix_dir+"transformix","-def",input_points_file,"-out", output_dir,"-tp",os.path.join(output_dir,"TransformParameters.1.txt")]
    transformix_cmd2  = [elastix_dir+"transformix","-in",annotationImagePath,"-out", output_dir,"-tp",os.path.join(output_dir,"TransformParameters.1.txt")]


    #Adjust Scale
    mImage = nib.load(moving_image)
    mData = mImage.get_fdata()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        subprocess.run(registration_cmd) 

    if args.cell_detection:

        if not os.path.isfile(input_points_file):
            if args.input_mat:
                cell_locations =get_input_cell_locations(input_mat)
                cell_locations = cell_locations[:, [2,1,0]]
                
                cell_locations[:,1] = mData.shape[1] - cell_locations[:,1]-1
                cell_locations[:,2] = mData.shape[2] - cell_locations[:,2]-1
            else:
                imgfiles = natsorted(glob.glob(img_dir+"/**/*1_{}.tif".format(channel), recursive=True))
                #print(img_files)
                cells = Parallel(n_jobs=-4, verbose=13)(delayed(get_cell_locations)(img_file, index =i, intensity_threshold=threshold) for i, img_file in enumerate(imgfiles))
                cell_locations = np.vstack(cells)
                cell_locations[:,0] = mData.shape[0] - cell_locations[:,0]-1

            
            points = cell_locations
            createShardedPointAnnotation(points,img_dir )

            scaledCellLocations = np.round(cell_locations*[ 1, 1/20,1/20]).astype(int)
            np.savetxt(input_points_file, scaledCellLocations , "%d %d %d", header = "index\n"+str(cell_locations.shape[0]), comments ="")
        

        subprocess.run(transformix_cmd)
        scaledCellLocations = parsePhysicalPointsFromOutputFile(os.path.join(output_dir,"outputpoints.txt"))
        #np.savetxt(output_points_file, scaledCellLocations , "%d %d %d", header = "index\n"+str(scaledCellLocations.shape[0]), comments ="")
        annotationImage  = sitk.ReadImage(annotationImagePath)
        outputIndices = convertPhysicalPointsToIndex(scaledCellLocations , annotationImage)

        if args.flag:
            outputIndices = np.loadtxt(input_points_file , skiprows=2)
            replaceBSplineOrder(os.path.join(output_dir,"TransformParameters.1.txt"))
            subprocess.run(transformix_cmd2)
            annotationImage  = sitk.ReadImage(os.path.join(output_dir,"result.nii"))

        cellRegionCounts, pointIndices = countCellsInRegions( outputIndices, annotationImage)
        pd.DataFrame(dict(cellRegionCounts).items(), columns=["region", "count"]).to_csv(cell_count_file, index=False)
        atlas_df = pd.read_csv(ATLAS_PATH, index_col=None)
        count_df = pd.read_csv(cell_count_file, index_col=None)
        region_df,count_df = process_counts(atlas_df, count_df)
        count_df.to_csv(os.path.join(output_dir,"cell_region_count.csv"), index=False)
        region_df.to_csv(os.path.join(output_dir,"region_counts.csv"), index=False)