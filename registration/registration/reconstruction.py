#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atchuth Naveen
Code developed at UC Irvine.
3D volume reconstruction utilities
"""

import glob
import os

import numpy as np

import SimpleITK as sitk
from skimage.transform import resize
from natsort import natsorted
from joblib import Parallel, delayed

from registration.utils import create_nifti_image

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
