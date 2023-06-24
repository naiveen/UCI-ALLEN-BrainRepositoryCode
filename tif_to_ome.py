#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
command line executable to convert a directory of tif images
to zarr format that can be visualized on neuroglancer

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
from skimage import io, exposure, data

from natsort import natsorted
from joblib import Parallel, delayed

import zarr
from dask.array import from_zarr
from ome_zarr.writer import write_image
from ome_zarr.io import parse_url


def arg_parser():
    parser = argparse.ArgumentParser(description='merge 2d tif images into a 3d image')
    parser.add_argument('img_dir', type=str,
                        help='path to tiff image directory')
    parser.add_argument('out_dir', type=str,
                        help='path to output the corresponding tif image slices')
    parser.add_argument('--channel', type=int, default =0,
                        help='Channel to generate nii file. 0 is default signal channel.')

    return parser


"""
def read_image( file_name):
    image = sitk.ReadImage(str(file_name))
    image = np.rot90(sitk.GetArrayFromImage(image)).astype(np.uint16)
    return image
    percentiles = np.percentile(image, (5, 95))
    # array([ 1., 28.])
    scaled = exposure.rescale_intensity(image,in_range=tuple(percentiles))
    #im1 = resize(image,(8192,8192),preserve_range=True)
"""

def read_image(i, zarr_file, file_name):
    image = sitk.ReadImage(str(file_name))
    image = sitk.GetArrayFromImage(image)
    image[image<0] = 0
    image = np.rot90(image).astype(np.uint16)
    percentiles = np.percentile(image, (0.5, 99.5))
    scaled = image
    scaled = exposure.rescale_intensity(image,in_range=tuple(percentiles))
    zarr_file[i,:,:]= scaled



def main():
    try:
        args = arg_parser().parse_args()
        img_dir = Path(args.img_dir)
        channel = args.channel
        """
        List of image files that needs to be stacked.
        """
        fns = natsorted(glob.glob(args.img_dir+"/**/*1_{}.tif".format(channel), recursive=True))
        print(len(fns))
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)

        """
        Path to zarr group
        """
        zarr_path  = os.path.join(args.out_dir,"data.zarr")
        """
        Create an ome zarr group
        """
        ome_path =os.path.join(args.out_dir,"ome.zarr")
        image = sitk.ReadImage(str(fns[0]))
        image = np.rot90(sitk.GetArrayFromImage(image))
        w,h = image.shape

        """
        Write into zarr group
        """
        zarr_file = zarr.open(zarr_path,shape= (len(fns),w,h), chunks = (1,w,h),  dtype='u2')
        Parallel(n_jobs=5, verbose=13)(delayed(read_image)(i,zarr_file, fn) for i,fn in enumerate(fns))
        print(zarr_file)
        """
        Read into a dask array
        """
        dask_arr = from_zarr(zarr_file)
        print(dask_arr.shape)

        """
        Write to a ome Zarr group
        """
        store = parse_url(ome_path, mode="w").store
        zarr_grp = zarr.open(store=store)
        z =1
        w=500
        h = 500
        write_image(dask_arr, group = zarr_grp,axes="zyx",storage_options=dict(chunks=(z, w, h)))
        print("OME Done")
        return 0
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())


"""

#uicontrol invlerp normalized(range=[0, 32667])
void main() {
      if(int(getDataValue(0).value)>0){
        emitRGB(normalized()*vec3(0, 1, 0));
}    else{
emitRGB(vec3(0, 0, 0));}
}
"""