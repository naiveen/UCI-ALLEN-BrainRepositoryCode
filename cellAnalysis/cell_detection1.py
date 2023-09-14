#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atchuth Naveen
Command line executable to detect cells and view them on neuroglancer window
to zarr format that can be visualized on neuroglancer

call as: python tif_to_nii.py img_dir out_dir {--channel channel}
(append optional arguments to the call as desired)
"""
import sys
import os
import webbrowser
import cv2
import glob
import argparse
from pathlib import Path
import json
import struct

import numpy as np
import scipy
import matplotlib.pyplot as plt

from natsort import natsorted

from skimage.feature import peak_local_max
from skimage.transform import resize
from skimage import io, exposure, data
from skimage.segmentation import watershed

import SimpleITK as sitk

from joblib import Parallel, delayed

import StructureElement as se
from ng_utils import *
def arg_parser():
    parser = argparse.ArgumentParser(description='Cell Counting Argments')

    parser.add_argument('--img_path', type=str, default =".",
                        help='path to tiff image')

    parser.add_argument('--img_dir', type=str,
                        help='path to tiff image directory')

    parser.add_argument('--out_dir', type=str, default="./",
                        help='path to output directory where cell counts file is stored.')
    
    parser.add_argument('--channel', type=int, default =0,
                        help='Channel to generate nii file. 0 is default signal channel.')

    parser.add_argument('--section', type=int, default =-1,
                        help='Section Number. If None, cells will be detected in all cells.')

    parser.add_argument('--threshold', type=int, default =10,
                        help='Background intensity threshold. Image is assumed to be in uint8. ')
    parser.add_argument('--size', type=int, default =10,
                        help='Size threshold. ')

    parser.add_argument('--min_distance', type=int, default =5,
                        help='Size threshold. ')


    parser.add_argument('--visualize',
                        help='To view detected cells on neuroglancer')

    return parser


def createShardedPointAnnotation(points, output_directory ):
    """
    points: List of points to write into Annotation layer
    output_directory: Output Directory to write the sharded annotation

    """
    output_directory = output_directory if isinstance(output_directory, os.PathLike) else Path(output_directory)
    points_directory = output_directory/"points"
    info = {
        "@type": "neuroglancer_annotations_v1",
        "annotation_type": "POINT",
        "by_id": { "key": "by_id" },
        "dimensions": {
            "z": [50,"m"],
            "x": [1.25,"m"],
            "y": [1.25,"m"]
        },
        "lower_bound": [0,0,0],
        "upper_bound": [300, 12000,8000],
        "properties": [],
        "relationships": [],
        "spatial": [
            {
                "chunk_size": (points.max(0) + 1).astype(int).tolist(),
                "grid_shape": [1, 1, 1],
                "key": "spatial0",
                "limit": len(points)+1
            }
        ]
    }


    info_path = points_directory / "info"
    if not info_path.parent.exists():
        info_path.parent.mkdir()
    with info_path.open("w") as fd:
        json.dump(info, fd, indent=2)

    # Yet to do : Multi Sharded Annotation
    spatial_path = points_directory / "spatial0" / "0_0_0"
    if not spatial_path.parent.exists():
        spatial_path.parent.mkdir()
    with spatial_path.open('wb') as fd:
        total_points = len(points)
        buffer = struct.pack('<Q', total_points)
        for (x, y, z) in points:
            annotpoint = struct.pack('<3f', x,y,z)
            buffer += annotpoint
        pointid_buffer = struct.pack('<%sQ' % len(points), *range(len(points)))
        buffer += pointid_buffer
        fd.write(buffer)


def readSectionTif( file_path):
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


def remove_background(source, shape =(10,10), form = 'Disk'):
    """
    Performs Morpological Background removal.
    source : 2D np array representing an image.
    """
    selem = se.structure_element(shape, form=form, ndim=2).astype('uint8');
    
    morph = np.minimum(source, cv2.morphologyEx(source, cv2.MORPH_OPEN, selem))
    removed = source - morph
    removed -= np.array(source>200)  *removed
    removed += np.array(source>200)  *source
    return removed


def get_cell_locations(img, intensity_threshold =5,  min_distance = 5, size_threshold =15, index = None):
    """
    img : 2d Image array or img_path
    intensity_threshold: backgound intensity in 0 to 255 scale
    size_threshold : Minimum cell size
    min_distance: Minimum distance between two cell centers
    
    index : Cell z index
    """
    
    if ( type(img) == str):
        img = readSectionTif(img)
    
    if(img.shape[0] ==3):
        img = np.squeeze(img[2,:,:])
    section = (img/np.iinfo(img.dtype).max)*255

    section = section.astype(np.uint8)
    #print("Background removal")
    bg =remove_background(section)
    print("Peak Detection")
    peaks  = peak_local_max(bg,min_distance =min_distance,threshold_abs =intensity_threshold)

    mask = np.zeros(bg.shape)
    mask[tuple(peaks.T)] = True
    markers, _ = scipy.ndimage.label(mask)
    print("Size Detection")
    labels = watershed(-bg, markers, mask = bg> intensity_threshold);

    centers = np.array(scipy.ndimage.center_of_mass(bg, labels, index=np.arange(1, np.max(labels)+1)))
    centers = centers.astype(int)

    sizes = scipy.ndimage.sum(np.ones(labels.shape, dtype=bool), labels=labels, index=np.arange(1, np.max(labels)+ 1));
    cells = centers[sizes>=size_threshold]
    
    if index is not None:
        cells = np.hstack([np.zeros((cells.shape[0],1))+ index, cells])
    
    return cells, img

def main():
    args = arg_parser().parse_args()
    
    imgfile  = args.img_path
    threshold  = args.threshold
    if args.img_path ==".":
        img_dir = Path(args.img_dir)
        channel = args.channel
        section = args.section


        input_points_file = img_dir/"inputpoints.txt"

        imgfiles = natsorted(glob.glob(os.path.join(img_dir,"**/*1_{}.tif".format(channel)), recursive=True))

        cells = Parallel(n_jobs=-4, verbose=13)(delayed(get_cell_locations)(img_file, intensity_threshold=threshold, index =i) for i, img_file in enumerate(imgfiles))
        points = np.vstack(cells)
        np.savetxt(input_points_file, points , "%d %d %d", header = "point\n"+str(cells.shape[0]), comments ="")
        createShardedPointAnnotation(points,img_dir)
    else:
        image = readSectionTif(imgfile)
        print("Detecting Cells")
        cells, image= get_cell_locations(image, intensity_threshold=threshold, min_distance = args.min_distance , size_threshold = args.size)
        viewer = ng_SingleSectionLocalViewer(image, cells)
        webbrowser.open(viewer.get_viewer_url())
        input("Press any key to continue")
        while(input("Press x to exit, any key to reload......")!= "x"):
            viewer = ng_SingleSectionLocalViewer(image, cells, viewer)
            webbrowser.open(viewer.get_viewer_url())
        

if __name__ == "__main__":
    sys.exit(main())