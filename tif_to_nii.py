#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tif_to_nii

command line executable to convert a directory of tif images
to a nifti image stacked along a user-specified axis

call as: python tif_to_nii.py img_dir out_dir {--channel channel}
(append optional arguments to the call as desired)

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

def arg_parser():
    parser = argparse.ArgumentParser(description='merge 2d tif images into a 3d image')
    parser.add_argument('img_dir', type=str,
                        help='path to tiff image directory')
    parser.add_argument('out_dir', type=str,
                        help='path to output the corresponding tif image slices')

    parser.add_argument('--channel', type=int, default =0,
                        help='Channel to generate nii file. 0 is default signal channel.')
    return parser


def read_image(i, file_name):
    """
    All images are being downscaled by 8 because CCF max template size is 10um. 
    Resolution on x and y axes = 1.25 um approximately.

    i - index for parallel processing
    file_name - file name to open
    """
    image = sitk.ReadImage(str(file_name))
    #image = np.rot90(sitk.GetArrayFromImage(image))
    image = sitk.GetArrayFromImage(image)
    #im1 = resize(image,(image.shape[0]/20, image.shape[1]/20),preserve_range=True)
    im = resize(image, (image.shape[0]/8, image.shape[1]/8),preserve_range=True)
    del image
    return [i,im]

def get_stacked_data(img_array):
    """
    Reorient stacked image data to match CCF format
    """
    img_data  = np.stack(img_array,axis=0)
    img_data = img_data.transpose([0,2,1])
    img_data = np.flip(img_data,axis=0)
    img_data = np.flip(img_data,axis=1)
    img_data = np.flip(img_data,axis=2)
    return img_data


def create_nifti_image(img_array, scale, output_dir):
    """
    img_array : numpy array, containing stack of images
    scale: nifti scale
    """
    affine_transform = np.zeros((4,4))

    affine_transform[0,2] = 0.01 * scale
    affine_transform[2,1] = -0.01* scale
    affine_transform[1,0] = -0.05
    affine_transform[3,3] = 1
    nibImg = nib.Nifti1Image(img_array,affine_transform)

    nibImg.header['qform_code'] =1
    nibImg.to_filename(os.path.join(output_dir, 'brain_{}.nii.gz'.format(scale*10)))



def main():
    try:
        args = arg_parser().parse_args()

        img_dir = Path(args.img_dir)

        channel = args.channel
        fns = natsorted(glob.glob(args.img_dir+"/**/*1_{}.tif".format(channel), recursive=True))
        print(len(fns))
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)


        img_list  =  Parallel(n_jobs=-2, verbose=13)(delayed(read_image)(i, fn) for i,fn in enumerate(fns))
        img_list  = sorted(img_list)

        img_list = [img for _,img in img_list]


        for scale in [1,2.5, 5, 10]:

            im_list =[resize(img, (img.shape[0]/scale, img.shape[1]/scale),preserve_range=True) for img in img_list]
            
            img_array = get_stacked_data(im_list)
            create_nifti_image(img_array, scale, args.out_dir)

        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())