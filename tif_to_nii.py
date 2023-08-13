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
from reconstruction import *

def arg_parser():
    parser = argparse.ArgumentParser(description='merge 2d tif images into a 3d image')
    parser.add_argument('img_dir', type=str,
                        help='path to tiff image directory')
    parser.add_argument('out_dir', type=str,
                        help='path to output the corresponding tif image slices')
    parser.add_argument('--channel', type=int, default =1,
                        help='Channel to generate nii file. 0 is default signal channel.')
    return parser


def main():
    try:
        args = arg_parser().parse_args()

        createNiiImages(args.img_dir,args.out_dir, args.channel)

        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())