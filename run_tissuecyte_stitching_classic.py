"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
Atchuth Naveen
Code developed at UC Irvine.

Corrects deformation in individual images and stitches them into a large mosaic section. 
"""


import sys
import argparse
import logging
import os

from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom

import SimpleITK as sitk
import numpy as np
from six import iteritems

from stitcher import Stitcher
from tile import Tile
import json

import itertools
import cv2
import glob
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
logging.getLogger().setLevel(logging.INFO)
logging.captureWarnings(True)

import numpy as np
import scipy.stats
from scipy.special import binom
from skimage.transform import resize

import sys
joblib_backend = None
if sys.platform == 'win32':
    joblib_backend = 'multiprocessing'



n_threads  = -3


def bernstein(u, n, k):
    return binom(n,k) * u**k * (1-u) **(n-k)

def barray(u,v, n,m):
    bmatrix =np.zeros((len(u), (n+1)*(m+1)))
    for i in range(n+1):
        for j in range(m+1):
            bmatrix[:,i*(m+1) + j] = bernstein(u,n,i)* bernstein(v,m,j)
            #blist.append(bernstein(u,n,i)* bernstein(v,m,j))
    return bmatrix




def create_perfect_grid(nhs, nvs, lw,sw):
    """
    nhs -  number of horizantal square
    nvs - number of vertical squares
    lw - line width
    sw - square width
    """
    xs =20
    ys= 20
    
    im = np.zeros((2*xs+ nvs*sw +lw, 2*ys+ nhs*sw + lw))
    for i in range(nhs+1):
        cv2.line(im, (xs+i*sw,ys), (xs+i*sw, im.shape[0]-ys-int(lw/2)), (255, 255, 255), thickness=lw)
           
    for i in range(nvs+1):
        cv2.line(im, (xs, ys+i*sw), (im.shape[1] -xs-int(lw/2), ys + i*sw), (255, 255, 255), thickness=lw) 
    return im


def get_deformation_map(width, height, kx, ky):
    px=[]
    py=[]
    for i in range(2*width):
        for j in range(2*height):
            px.append(j)
            py.append(i)
    px = np.asarray(px)
    py = np.asarray(py)
    px_ = np.asarray(px/(2*float(height)))
    py_ = np.asarray(py/(2*float(width)))

    pX_ = np.matmul(barray(px_,py_,4,4),kx)
    pY_ = np.matmul(barray(px_,py_,4,4),ky)
    pX_ = pX_ * height
    pY_ = pY_ * width
    
    pX_[pX_<=0] = 0
    pX_[pX_>=height-1] = height-1
    pY_[pY_<=0] = 0
    pY_[pY_>=width-1] = width -1
    
    return pX_,pY_


corners1 = np.asarray([[33,10], [796,21],[30,813], [793,818]])
corners2 = np.asarray([[20,20], [776,20],[20,794], [776,794]])
H, _ = cv2.findHomography(corners1, corners2)
gridp = create_perfect_grid(42,43, 4,18)
gridp = gridp[20:794,20:776]

kx,ky = joblib.load("bezier16x.pkl" )

#Double the size to preserve sampling , need to downsample later
pX_,pY_ = get_deformation_map(gridp.shape[0],gridp.shape[1], kx, ky)



def correct_deformation(im0,H,pX_,pY_):
    im_warp = cv2.warpPerspective(im0, H, (im0.shape[1], im0.shape[0]))
    im_warp = im_warp[20:794,20:776]
    h,w  = im_warp.shape
    
    im = np.zeros(pX_.shape).astype(np.float64)
    x1 = np.floor(pX_).astype(int)
    x2 = np.ceil(pX_).astype(int)
    y1 = np.floor(pY_).astype(int)
    y2 = np.ceil(pY_).astype(int)
    
    dx1 = pX_ - x1
    dx2 = x2 - pX_
    dy1 = pY_ - y1
    dy2 = y2 - pY_
    dx1[np.where(y1==y2)] =1
    dy1[np.where(x1==x2)] =1
    
    im_warp1d = im_warp.ravel()
    im = im_warp1d[y1 * w+ x1]* dx1*dy1+ im_warp1d[y1*w+ x2]*dx2*dy1+im_warp1d[y2 * w+ x1]*dy2*dx1+im_warp1d[y2 * w+ x2]*dy2*dx2
    
    im = np.reshape(im,(2*h,2*w))
    im =resize(im,(h,w),preserve_range=True)
    return im

def get_missing_tile_paths(missing_tiles):

    paths = []

    for index, path in iteritems(missing_tiles):
        spath = ','.join(map(str, path))
        logging.info('writing missing tile path for tile {0} as {1}'.format(index, spath))
        paths.append(spath)

    return paths


def read_image(file_name):
    image = sitk.ReadImage(str(file_name))
    return sitk.GetArrayFromImage(image)
    #return np.flipud(sitk.GetArrayFromImage(image)).T

def write_output(imgarr, path):
    imgarr[imgarr<0] =0
    image = sitk.GetImageFromArray(imgarr)
    image = sitk.Cast(image, sitk.sitkUInt16)
    sitk.WriteImage(image,path)


def normalize_image_by_median(image):

    median = np.median(image)

    if median != 0:
        image = np.divide(median, image)
        image[np.isnan(image)] = 0
        image[np.isinf(image)] = 0

    return image


def load_average_tile(path):
    tile = read_image(path)
    return normalize_image_by_median(tile)


def get_section_avg(tiles):
    imlist=[[],[],[],[]]
    avg_tiles =[]
    for tile in tiles:
        try:
            im = read_image(tile['path'])
            #im = cv2.resize(im, (832,832))
            imlist[tile["channel"]-1].append(im)
        except(IOError, OSError, RuntimeError) as err:
            logging.info('did not find image tile for channel {0} (zero-indexed)'.format(tile["channel"]-1))
    avg_tiles.append(np.mean(imlist[0],axis=0))
    avg_tiles.append(np.mean(imlist[1],axis=0))
    avg_tiles.append(np.mean(imlist[2],axis=0))
    avg_tiles.append(np.mean(imlist[3],axis=0))
    return avg_tiles

def generate_avg_tiles(section_jsons, avg_tiles_dir):
    if not os.path.isdir(avg_tiles_dir):
        os.mkdir(avg_tiles_dir)    
    logging.info('Generating Avg Tiles.')
    imlist = Parallel(n_jobs=n_threads, verbose=13)(delayed(get_section_avg)(section_json['tiles']) for section_json in section_jsons)
    imlist = np.asarray(imlist)
    avg_tiles = np.mean(imlist, axis=0)
    for i,im in enumerate(avg_tiles):
        image = sitk.GetImageFromArray(im)
        image = sitk.Cast(image, sitk.sitkFloat32)
        sitk.WriteImage(image, os.path.join(avg_tiles_dir , "avg_tile_"+str(i)+".tif"))


def generate_tiles(tiles, avg_tiles, output_dir, save_undistorted= False):
    tile_obj_list =[]

    for tile_params in tiles:
        tile = tile_params.copy()
        try:
            im = read_image(tile['path'])
            #im = cv2.resize(im, (832,832))
            im = np.multiply(im, avg_tiles[tile['channel']-1] )
            im_corrected = correct_deformation(im, H, pX_, pY_)
            tile['image'] = im_corrected
            if save_undistorted:
                undistorted_tile_path = os.path.join(undistorted_dir, "ch{}".format(tile['channel']-1), os.path.split(tile['path'])[1])
                write_output(np.ascontiguousarray(im_corrected), undistorted_tile_path)
            tile['is_missing'] = False
        except (IOError, OSError, RuntimeError) as err:
            tile['image'] = None
            tile['is_missing'] = True
        
        tile['channel'] = tile['channel'] - 1
        tile_obj = Tile(**tile)
        del tile
        #tile_obj_list.append(tile_obj)
        yield tile_obj
    




def create_section_json(root_dir, sno ,sectionName,  mosaic_data, depth =0):
    tyx = -3
    tyy = -43
    txx = -25
    txy = 5
    margins= {"row": 0,"column": 0}
    size= {"row": 774,"column": 756}
    startx = 200
    starty = 200

    
    mcolumns = int(mosaic_data["mcolumns"])
    mrows = int(mosaic_data["mrows"])
    index_ = (sno*int(mosaic_data["layers"])+depth)*mrows*mcolumns 
    image_dimensions= {"row": mrows*size['row'] + 2*startx ,"column": mcolumns*size['column'] + 2*starty}
    
    section_json ={}
    section_json["mosaic_parameters"]= mosaic_data
    tiles =[]
    for ncol in range(mcolumns):
        for nrow in range(mrows):
            index = index_ + (ncol)*mrows+nrow
            #index = sno*mrows*mcolumns + (ncol)*mrows+nrow
            tile_paths = glob.glob(f'{sectionName}/*-{index}_*.tif')
            #print(f'{sectionName}/*-{index}_*.tif', tile_paths, sectionName, index)
            if(len(tile_paths) ==0):
                continue
            bounds = {}
            row ={}
            col ={}
            if(ncol%2 ==0):
                row["start"] = starty + nrow* size["row"] + nrow * tyy + ncol * txy 
                row["end"] = row["start"] + size["row"]
                col["start"] = startx + ncol * size["column"] + ncol * txx+ nrow* tyx 
                col["end"] = col["start"] + size["column"]
            else:
                row["start"] = starty + (mrows - nrow-1)* size["row"] + (mrows - nrow-1) * tyy + ncol * txy 
                row["end"] = row["start"] + size["row"]
                col["start"] = startx + ncol * size["column"] + ncol * txx+ (mrows - nrow-1)* tyx 
                col["end"] = col["start"] + size["column"]
            bounds["row"] = row
            bounds["column"] = col
            for ch, path in enumerate(tile_paths):
                tile_data ={}
                tile_data["path"] = path
                tile_data["bounds"] = bounds
                tile_data["margins"]= margins
                tile_data["size"] = size
                tile_data["channel"] = ch+1
                tile_data["index"] = index
                tiles.append(tile_data)
        #index_ = index_ + mrows*mcolumns*int(mosaic_data["layers"])
        section_json["channels"] = list(range(1,int(mosaic_data["channels"])+1))
        section_json["tiles"] = tiles
        section_json['slice_fname'] = os.path.split(sectionName)[-1]+"_"+str(depth+1)
        section_json["image_dimensions"] = image_dimensions
    return section_json


def get_section_data(root_dir, depth=1, sectionNum=0):
    
    files = glob.glob(root_dir+'Mosaic*')
    print(root_dir, files)
    if(len(files) ==0 ):
        logging.info("Mosaic File Missing.")
    else:
        mosaic_file = files[0]

    mosaic_data ={}
    with open(mosaic_file) as fp:
        for line in fp:
            k,v = line.rstrip("\n").split(":",1)
            mosaic_data[k]=v

    sectionNames = glob.glob(root_dir+mosaic_data["Sample ID"]+"*")
    section_jsons_list =[]
    if(sectionNum!=0):
        sectionName  =  os.path.join(root_dir,"{}-{:04d}".format( mosaic_data["Sample ID"], sectionNum+1))
        section_jsons = [create_section_json(root_dir, sectionNum, sectionName, mosaic_data)]
        return mosaic_data, section_jsons
    for d in range(depth):
        section_jsons = Parallel(n_jobs=n_threads)(delayed(create_section_json)(root_dir,sno, sectionName, mosaic_data, d) for sno,sectionName in enumerate(sectionNames))
        section_jsons_list.append(section_jsons)
    section_jsons = list(itertools.chain.from_iterable(section_jsons_list))
    #print(section_jsons)

    return mosaic_data, section_jsons

def stitch_section(data, avg_tiles, output_dir, save_undistorted =False):
    
    tiles = generate_tiles(data['tiles'], average_tiles, output_dir, save_undistorted)
    stitcher = Stitcher(data['image_dimensions'], tiles, data['channels'])
    image, missing = stitcher.run()
    del tiles
    missing_tile_paths = get_missing_tile_paths(missing)

    for ch in range(image.shape[2]):
        slice_path = os.path.join(output_dir, "stitched_ch{}".format(ch), data['slice_fname']+"_{}.tif".format(ch))
        print(slice_path)
        write_output(np.ascontiguousarray(image[:,:,ch]), slice_path)

       

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--depth', default = 1, type=int)
    parser.add_argument('--sectionNum', default = 0, type=int)
    parser.add_argument('--save_undistorted', default=False, type=bool)
    args = parser.parse_args()

    root_dir = os.path.join(args.input_dir, '')
    output_dir = os.path.join(args.output_dir, '')
    depth = args.depth
    sectionNum = args.sectionNum
    save_undistorted = args.save_undistorted
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    print("Creating Stiching Json for Sections")
    mosaic_data, section_jsons = get_section_data(root_dir, depth, sectionNum)

    channel_count = int(mosaic_data['channels'])
    print("Creating Intermediate directories")
    for ch in range(channel_count):
        ch_dir  = os.path.join(output_dir, "stitched_ch{}".format(ch),"")
        if not os.path.isdir(ch_dir):
            os.mkdir(ch_dir)

    if save_undistorted:
        undistorted_dir = output_dir+"/undistorted"

        if not os.path.isdir(undistorted_dir):
            os.mkdir(undistorted_dir)

        for ch in range(channel_count):
            ch_dir = os.path.join(undistorted_dir, "ch{}".format(ch),"")
            if not os.path.isdir(ch_dir):
                os.mkdir(ch_dir)

    average_tiles =[]
    if(sectionNum==0):
        avg_tiles_dir = os.path.join(output_dir,"avg_tiles")
        print("Generating Avg tiles")

        generate_avg_tiles(section_jsons, avg_tiles_dir)
        for i in range(4):
            average_tiles.append(load_average_tile(os.path.join(avg_tiles_dir,"avg_tile_"+str(i)+".tif")))
    else:
        for i in range(4):
            average_tiles.append(np.ones((832,832)))
    print("Stitiching")
    #Parallel(n_jobs=1, backend=joblib_backend)(delayed(stitch_section)(section_json,average_tiles, output_dir) for section_json in tqdm(section_jsons))
    Parallel(n_jobs=n_threads, verbose=13)(delayed(stitch_section)(section_json,average_tiles, output_dir, save_undistorted) for section_json in section_jsons)
