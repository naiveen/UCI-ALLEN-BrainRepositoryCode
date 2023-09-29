#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atchuth Naveen
Code developed at UC Irvine.
3D volume reconstruction utilities
"""

import argparse
import glob
import os
from pathlib import Path
import sys

from PIL import Image
import nibabel as nib
import numpy as np

import SimpleITK as sitk
from tqdm import tqdm
from skimage.transform import resize
from natsort import natsorted
from joblib import Parallel, delayed

def readTifWrapper(i, file_name):
    """
    All images are being downscaled by 8 because CCF max template size is 10um. 
    Resolution on x and y axes = 1.25 um approximately.

    i - index for parallel processing
    file_name - file name to open
    """
    image = readTifSection(str(file_name))
    #image = np.rot90(sitk.GetArrayFromImage(image))
    #im1 = resize(image,(image.shape[0]/20, image.shape[1]/20),preserve_range=True)
    im = resize(image, (image.shape[0]/8, image.shape[1]/8),preserve_range=True)
    del image
    return [i,im]

def readTifSection( file_path):
    """
    Read tif section image using SITK
    """
    image = sitk.ReadImage(str(file_path))
    image = sitk.GetArrayFromImage(image)
    image[image<0] = 0
    image = image.T
    image = np.flip(image,axis=0)
    image = np.flip(image,axis=1)
    image  = np.squeeze(image)
    return image


def get_stacked_data(img_array):
    """
    Reorient stacked image data to match CCF format
    """
    img_data  = np.stack(img_array,axis=0)
    img_data = np.flip(img_data,axis=0)
    return img_data

def create_nifti_image(img_array, scale, name=None, sz= None):
    """
    img_array : numpy array, containing stack of images
    scale: nifti scale
    """


    # The following parameters are set to be consistent with ALLEN CCF NII Templates
    affine_transform = np.zeros((4,4))
    affine_transform[0,2] = 0.01 * scale
    affine_transform[2,1] = -0.01* scale
    if sz==None:
        affine_transform[1,0] = -0.05
    else:
        affine_transform[1,0] = -0.05 *sz
    affine_transform[3,3] = 1
    nibImg = nib.Nifti1Image(img_array,affine_transform)
    nibImg.header['qform_code'] = 1
    nibImg.header['qoffset_x'] = -5.695
    nibImg.header['qoffset_y'] = 5.35
    nibImg.header['qoffset_z'] = 5.22


    if name != None:
        if name[-1]!='z':
            name  = os.path.join(name, 'brain_{}.nii.gz'.format(int(scale*10))) 
        nibImg.to_filename( name)
    return nibImg


def createNiiImages(img_dir, out_dir , channel=0):
    """
    Creates Nii format 3d volumes from a stack of tif files

    img_dir: Directory containing nii files
    out_dir: Directory to store nii files
    channel: Channel to use to create nii files
    """

    if out_dir == None:
        out_dir = os.path.join(img_dir,"nii" )

    fns = natsorted(glob.glob(img_dir+"/**/*1_{}.tif".format(channel), recursive=True))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    img_list  =  Parallel(n_jobs=-2, verbose=13)(delayed(readTifWrapper)(i, fn) for i,fn in enumerate(fns))
    img_list  = sorted(img_list)

    img_list = [img for _,img in img_list]


    for scale in [1,2.5, 5, 10]:

        im_list =[resize(img, (img.shape[0]/scale, img.shape[1]/scale),preserve_range=True) for img in img_list]
        
        img_array = get_stacked_data(im_list)
        create_nifti_image(img_array, scale, out_dir)
