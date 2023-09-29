"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
Atchuth Naveen
Code developed at UC Irvine.

Utility file to create NII image volumes, perform registration, cell counting, cell region detection. 
"""
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
from cellAnalysis.cell_counting import *
from cellAnalysis.cell_detection import *
from reconstruction import *
import registration as rg
import shutil
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform, geometric_transform
from registration.reconstruction import create_nifti_image
from registration.reg_utils import loadNiiImages
import registration as rg



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

def loadTransformMapping(fshape, mshape,regDir):
    """
    Returns the mapping of shape fshapex3. 
    For each voxel located at x,y,z in fdata, the 1x3 cell represents the x_,y_,z_ in mdata. 
    """
    
    mx, my, mz = mshape
    fx, fy, fz = fshape
    
    #Sample volumes that represent the coordinate of the voxel
    X, Y,Z= np.meshgrid(range(mx), range(my), range(mz), indexing = 'ij')
    X =X.astype(float)
    Y =Y.astype(float)
    Z= Z.astype(float)
    
    
    A = np.load(os.path.join(regDir,"axisAlignA.npy"))
    alignedX = affine_transform(X,np.linalg.inv(A), output_shape = X.shape, order =1)
    alignedY = affine_transform(Y,np.linalg.inv(A), output_shape = Y.shape, order =1)
    alignedZ = affine_transform(Z,np.linalg.inv(A), output_shape = Z.shape, order =1)
    alignedXPath = os.path.join(regDir, "alignX.nii.gz")
    alignedYPath = os.path.join(regDir, "alignY.nii.gz")
    alignedZPath = os.path.join(regDir, "alignZ.nii.gz")
    
    create_nifti_image(alignedX, 2.5, alignedXPath, 1)
    create_nifti_image(alignedY, 2.5, alignedYPath, 1)
    create_nifti_image(alignedZ, 2.5, alignedZPath, 1)

    elastixX =  rg.elastixTransformation(alignedXPath, regDir, os.path.join(regDir, "elastixX"))
    elastixY =  rg.elastixTransformation(alignedYPath, regDir, os.path.join(regDir, "elastixY"))
    elastixZ =  rg.elastixTransformation(alignedZPath, regDir, os.path.join(regDir, "elastixZ"))

    deformationField = np.load(os.path.join(regDir,"deformation3d.npy"))
    finalX   = rg.applyDeformationField(elastixX , deformationField)
    finalY   = rg.applyDeformationField(elastixY , deformationField)
    finalZ   = rg.applyDeformationField(elastixZ , deformationField)
    
    mapping = np.zeros((fx,fy,fz,3))
    mapping[:,:,:, 0] = finalX

    mapping[:,:,:, 1] = finalY

    mapping[:,:,:, 2] = finalZ
    mapping[mapping<0] = 0
    
    #Remove all junk files
    os.remove(os.path.join(regDir, "alignX.nii.gz"))
    os.remove(os.path.join(regDir, "alignY.nii.gz"))
    os.remove(os.path.join(regDir, "alignZ.nii.gz"))
    shutil.rmtree(os.path.join(regDir, "elastixX"))
    shutil.rmtree(os.path.join(regDir, "elastixY"))
    shutil.rmtree(os.path.join(regDir, "elastixZ"))
    
    return mapping      

def getMappedIndices(points, mapping):
    """
    To be used in conjunction with loadTransformMapping

    point - px3 np array p is the number of points

    mapping- nxwxhx3 np array
    
    """
    mint = mapping.astype(int)
    mappedIndices =[]
    for point in tqdm(points):
        try:
            x,y,z = point
            x_,y_,z_ = np.where(np.bitwise_and(np.bitwise_and(mint[:,:,:,0]==x, mint[:,:,:,1]==y), mint[:,:,:,2]==z))
            if(len(x_) ==0 ):
                mappedIndices.append([0,0,0])
                #mnorm  = np.linalg.norm(mapping - np.reshape(point, (1,1,1,3)), axis=3)
                #x_,y_,z_ = np.where(mnorm == np.min(mnorm))
            else:
                mappedIndices.append([x_[0],y_[0],z_[0]])
            #region_ids.append(stats.mode(anndata[indices[0], indices[1], indices[2]], keepdims=False).mode)
        except Exception as e:
            print("Exception at Mapping Indices for point {}".format(point))
    return mappedIndices

def registration(fixedImagePath, movingImagePath, outputDir):
    """
    Wrapper around different steps involved in Registration. 
    """
    print("Aligning Axes")
    A, axisAlignedData = rg.axisAlignData(fixedImagePath, movingImagePath)
    np.save(os.path.join(outputDir,"axisAlignA.npy"), A)
    axisAlignedDataPath  = os.path.join(outputDir , "axisAlignedData.nii.gz")
    create_nifti_image(axisAlignedData, 2.5, axisAlignedDataPath, 1)
    movingImagePath = axisAlignedDataPath

    print("Elastix Registration")
    elastixResult  = rg.elastixRegistration(fixedImagePath , movingImagePath, outputDir, rescale=True)
    elastixResult  = rg.elastixTransformation(axisAlignedDataPath, outputDir)
    movingImagePath = elastixResult

    print("Laplacian Refinement")
    deformationField  = rg.sliceToSlice3DLaplacian(fixedImagePath , movingImagePath , axis =0 )
    np.save(os.path.join(outputDir,"deformation3d.npy"), deformationField)
    transformedData   = rg.applyDeformationField(movingImagePath , deformationField)
    refinedResultPath = os.path.join(outputDir, "elastixRefined.nii.gz")
    create_nifti_image(transformedData, 2.5, refinedResultPath, 1/2)

    print("Final registered image stored at {}".format(refinedResultPath))


def resampleRegisteredSection(imgFileList, output_dir):
    """
    When imaged data is being registered to template, data needs to be resampled for cell detection. 

    Functionality to be added soon.
    """
    pass

if __name__ == '__main__':
    """
    This script can generate nii files and perform registration as well as use existing nii files for registration. 
    
    If --img_dir is provided, the script creates nii files according to the channel specified and use them for registration.

    If --img_dir is not provided, then script takes additional arguments --fixed_image, --moving_image. 

    In addition output_dir can be specified to store all intermediary and final results. 

    For performing cell detection and cell counting, additional arguments '--cell_detection' needs to be given. For cell detection, '--img_dir' is a required argument. 
    Threshold can be adjusted for cell counting with argument '--threshold'

    --input_mat is only present for legacy reasons and can be ignored. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default="", help = "Path to image directory containing stitiched sections")
    parser.add_argument('--channel', type=int, default=0, help = "Channel to use for registration and cell counting")
    parser.add_argument('--cell_detection', action='store_true', help ="Use this argument to do cell counting")
    parser.add_argument('--threshold', type=int, default=10, help ="Threshold for cell counting")

    parser.add_argument('--fixed_image', type=str, default = "") #.\reference\average_template_25.nii
    parser.add_argument('--moving_image', type=str, default ="") #output\brain25\result.1.nii
    parser.add_argument('--output_dir', type=str, default = "")
    parser.add_argument('--input_mat', type=str, default = None)

    parser.add_argument('--t2d', action='store_true',  help ="When template being registered to imaged brain volume and the resulting registration used for cell counting.") 

    args = parser.parse_args()
    elastix_dir = "./elastix/"
    template_dir = "./CCF_DATA/"
    annotationImagePath = r"./CCF_DATA/annotation_25.nii.gz"
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
        fixed_image = os.path.join(template_dir, "average_template_25.nii.gz")

    if args.output_dir=="":
        output_dir = os.path.join(img_dir, "reg")


    start = time.time()

    if not os.path.isdir(nii_dir):
        print("Creating Nii Images.")
        createNiiImages(img_dir, nii_dir, channel)
        print("Nii images created in {}".format(time.time()- start))
    else:
        print("Skipping Nii Image creation. Nii directory already present.")

    input_points_file = os.path.join(output_dir, "inputpoints.txt")
    output_points_file = os.path.join(output_dir, "output_points.txt")
    cell_count_file = os.path.join(output_dir, "cell_count.csv")
    
    if args.t2d:
        registration_cmd = [elastix_dir+"elastix","-m",fixed_image, "-f", moving_image, "-out", output_dir, "-p" , "001_parameters_Rigid.txt", "-p", "002_parameters_BSpline.txt"]
        transformix_cmd  = [elastix_dir+"transformix","-def",input_points_file,"-out", output_dir,"-tp",os.path.join(output_dir,"TransformParameters.1.txt")]
        transformix_cmd2 = [elastix_dir+"transformix","-in",annotationImagePath,"-out", output_dir,"-tp",os.path.join(output_dir,"TransformParameters.1.txt")]

    #Adjust Scale
    mImage = nib.load(moving_image)
    mData = mImage.get_fdata()

    fImage = nib.load(fixed_image)
    fData = fImage.get_fdata()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        if args.t2d:
            subprocess.run(registration_cmd)
        else:
            registration(fixed_image, moving_image, output_dir)

        print("Registrattion Done in {}".format(time.time()- start))
    else:
        print("Skipping Registration. registration directory already present.")

    if args.cell_detection:

        if not os.path.isfile(input_points_file):
            if args.input_mat:
                cell_locations =get_input_cell_locations(input_mat)
                cell_locations = cell_locations[:, [2,1,0]]
                
                cell_locations[:,1] = mData.shape[1] - cell_locations[:,1]-1
                cell_locations[:,2] = mData.shape[2] - cell_locations[:,2]-1
            else:
                imgfiles = natsorted(glob.glob(img_dir+"/**/*1_{}.tif".format(channel), recursive=True))
                
                cells = Parallel(n_jobs=-4, verbose=13)(delayed(get_cell_locations)(img_file, index =i, intensity_threshold=threshold) for i, img_file in enumerate(imgfiles))
                cell_locations = np.vstack(cells)
                cell_locations[:,0] = mData.shape[0] - cell_locations[:,0]-1

            points = cell_locations
            createShardedPointAnnotation(points,img_dir)
            scaledCellLocations = np.round(cell_locations*[ 1, 1/20,1/20]).astype(int)
            np.savetxt(input_points_file, scaledCellLocations , "%d %d %d", header = "index\n"+str(cell_locations.shape[0]), comments ="")
            print("Cell Detection done in {}".format(time.time()- start))
        else:
            scaledCellLocations = np.loadtxt(input_points_file, skiprows=2)
            print("Skipping Cell Detection. Cell Location file  already present.")

        """
        #Two methods for finding the region where the cell is located. Both of these methods give same results
        #Method1 : Transform cell locations in data space to the brain image space.
        #Method2 : Transform the annotation map to the brain image space.
        """
        annotationImage  = sitk.ReadImage(annotationImagePath)

        # The below transformatio needed only in case of registering template to data
        if args.t2d:
            subprocess.run(transformix_cmd)   #transforms points to annotation image space
            scaledCellLocations = parsePhysicalPointsFromOutputFile(os.path.join(output_dir,"outputpoints.txt"))
            outputIndices    = convertPhysicalPointsToIndex(scaledCellLocations , annotationImage)
            """
            #Method 2: Transform the annotation map to the brain image space
            outputIndices = np.loadtxt(input_points_file , skiprows=2)
            replaceBSplineOrder(os.path.join(output_dir,"TransformParameters.1.txt"))
            subprocess.run(transformix_cmd2)
            annotationImage  = sitk.ReadImage(os.path.join(output_dir,"result.nii"))
            """
        else:
            #mapping = loadTransformMapping(fData.shape, mData.shape, output_dir)
            #np.save("mapping.npy", mapping)
            mapping=np.load("mapping.npy")
            outputIndices = getMappedIndices(scaledCellLocations, mapping)

        print("Cell Location Transformation done in {}".format(time.time()- start))

        #np.savetxt(output_points_file, scaledCellLocations , "%d %d %d", header = "index\n"+str(scaledCellLocations.shape[0]), comments ="")
        cellRegionCounts, pointIndices = countCellsInRegions( outputIndices, annotationImage)
        pd.DataFrame(dict(cellRegionCounts).items(), columns=["region", "count"]).to_csv(cell_count_file, index=False)
        atlas_df = pd.read_csv(ATLAS_PATH, index_col=None)
        count_df = pd.read_csv(cell_count_file, index_col=None)
        region_df,count_df = process_counts(atlas_df, count_df)
        count_df.to_csv(os.path.join(output_dir,"cell_region_count.csv"), index=False)
        region_df.to_csv(os.path.join(output_dir,"region_counts.csv"), index=False)