#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atchuth Naveen
Contains Utilities to open a neuroglancer instance and feed data.
"""

import neuroglancer
import numpy as np
import imageio
import webbrowser
from time import sleep

res = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'],
        units=['um', 'um', 'um'],
        scales=[50, 1.25, 1.25])

def ng_createViewer():
    """
    Create a local neuroglancer Viewer
    """
    neuroglancer.set_static_content_source() 
    viewer=neuroglancer.Viewer()

    return viewer


def ng_SingleSectionLocalViewer(image, annotations=None, viewer=None):
    """
    image : 2D np array representing the image - gets expanded to 3D 
    annotations: nx2 np array of cell locations
    """

    def ngLayer(data,res,oo=[0,0,0],tt='segmentation'):
        return neuroglancer.LocalVolume(data,dimensions=res,volume_type=tt,voxel_offset=oo)
    if viewer is None:
        viewer = ng_createViewer()

    counter =0

    with viewer.txn() as s:
        s.layers.append(name='im',layer=ngLayer(np.expand_dims(image,axis=0),res,tt='image'))

    with viewer.txn() as s:
        s.crossSectionScale = 1
    if annotations is not None:
        with viewer.txn() as s:
            s.layers['annotation'] = neuroglancer.AnnotationLayer()
            ann = s.layers['annotation'].annotations
            # each point annotation has a unique id
            for x,y in annotations:
                pt = neuroglancer.PointAnnotation(point=[1,x, y], id=f'point{counter}')
                ann.append(pt)
                counter += 1
    return viewer


