#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atchuth Naveen
Code developed at UC Irvine.

command line executable to convert a directory of tif images
to zarr format that can be visualized on neuroglancer

call as: python tif_to_ome.py img_dir {--out_dir} {--channel channel}
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
from numcodecs import Blosc
import dask.array as da


def arg_parser():
    parser = argparse.ArgumentParser(description='tif images to ome-zarr format')
    parser.add_argument('--img_dir', type=str,
                        help='path to tiff image directory')
    parser.add_argument('--out_dir', default = "", type=str,
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

def readTifWrapper(i, zarr_file, file_name):
    """
    Wrapper around Tif reader to write directly into zarr file in parallel
    """
    image = readTifSection(str(file_name))
    #image = np.rot90(image).astype(np.uint16)
    #percentiles = np.percentile(image, (0.1, 99.9))
    scaled = image
    #scaled = exposure.rescale_intensity(image,in_range=tuple(percentiles))
    zarr_file[i,:,:]= scaled


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
        if args.out_dir == "":
            out_dir = img_dir
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        """
        Path to zarr group
        """
        zarr_path  = os.path.join(out_dir,"data.zarr")
        """
        Create an ome zarr group
        """
        ome_path =os.path.join(out_dir,"ome.zarr")
        image = readTifSection(str(fns[0]))
        w,h = image.shape

        """
        Write into zarr group
        """
        zarr_file = zarr.open(zarr_path,shape= (len(fns),w,h), chunks = (1,w,h),  dtype='u2')
        Parallel(n_jobs=5, verbose=13)(delayed(readTifWrapper)(i,zarr_file, fn) for i,fn in enumerate(fns))
        print(zarr_file)
        """
        Read into a dask array
        """
        dask_arr = from_zarr(zarr_file)
        print(dask_arr.shape)

        """
        Write to a ome Zarr group
        """
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        store = parse_url(ome_path, mode="w").store
        zarr_grp = zarr.open(store=store)
        z =5
        w=500
        h = 500
        #write_image(dask_arr, group = zarr_grp,axes="zxy",storage_options=dict(chunks=(z, w, h)))
        write_image(dask_arr, group = zarr_grp,axes="zyx",storage_options=dict(chunks=(z, w, h), compressor=compressor))

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