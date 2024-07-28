#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atchuth Naveen
Code developed at UC Irvine.
3D volume registration pipelines
"""

import numpy as np
import time
import registration.reg_utils as rg

from registration.utils import loadNiiImages
from registration.laplacianUtils import laplacianA1D, laplacianA3D
from scipy.sparse.linalg import lgmres
from tqdm import tqdm


def sliceToSlice2DLaplacian(fixedImage , movingImage ,sliceMatchList="same", axis=0):
    """
    Assumes both the images are matched slice to slice along axis='axis'
    Gets 2D correspondences between the slices and interpolates them smoothly to align both images
    """

    def nonLinearRegistrationWrapper(sno, fdata, mdata, axis =0):
        if sliceMatchList == "same":
            msno = sno
        else:
            msno= sliceMatchList[sno]
        dataimage = np.take(mdata, sno, axis=axis)
        templateimage = np.take(fdata, msno, axis=axis)
        fedge, medge, fbin, mbin = rg.getContours(templateimage, dataimage)
        dx, dy = rg.nonLinearRegistration2D(fedge, medge, fbin, mbin )
        return [np.zeros(dx.shape), dx, dy]

    fdata, mdata = loadNiiImages([fixedImage, movingImage])

    nx, ny, nz  = fdata.shape
    nd  = len(fdata.shape)
    
    deformationField = np.zeros((nd, nx, ny, nz))
    results =[]
    for sno in tqdm(range(fdata.shape[axis])):
        result = nonLinearRegistrationWrapper(sno, fdata, mdata , axis =axis)
        results.append(result)
    xstack =[x for x, y, z in results]
    ystack =[y for x, y, z in results]
    zstack =[z for x, y, z in results]
    
    dx = np.array(xstack)
    dy = np.array(ystack)
    dz = np.array(zstack)
    
    if axis ==0:
        deformationField[0] = dx
        deformationField[1] = dy
        deformationField[2] = dz
    
    if axis ==1:
        dx  = np.transpose(dx, axes = [1,0,2])
        dy  = np.transpose(dy, axes = [1,0,2])
        dz  = np.transpose(dz, axes = [1,0,2])
        deformationField[0] = dy
        deformationField[1] = dx
        deformationField[2] = dz
        
    if axis == 2:
        dx  = np.transpose(dx, axes = [1,2,0])
        dy  = np.transpose(dy, axes = [1,2,0])
        dz  = np.transpose(dz, axes = [1,2,0])
        deformationField[0] = dy
        deformationField[1] = dz
        deformationField[2] = dx
        
    return deformationField

def sliceToSlice3DLaplacian(fixedImage , movingImage ,sliceMatchList="same", axis  =0):
    """
    Assumes both the images are matched slice to slice according to sliceMatchList along axis- 'axis'
    Gets 2D correspondences between the slices and interpolates them smoothly across the volume
    """
    fdata, mdata = loadNiiImages([fixedImage, movingImage])
    
    nx, ny, nz  = fdata.shape
    nd  = len(fdata.shape)
    
    deformationField = np.zeros((nd, nx, ny, nz))
    
    flen  = nx*ny*nz
    Xcount = np.zeros(flen)
    Ycount = np.zeros(flen)
    Zcount = np.zeros(flen)

    Xd =  np.zeros(flen)
    Yd =  np.zeros(flen)
    Zd =  np.zeros(flen)
    
    flist =[]
    mlist =[]

    for sno in tqdm(range(fdata.shape[axis])):
        if sliceMatchList =="same":
            msno = sno
        else:
            msno = sliceMatchList[sno]
        dataimage = np.take(mdata, sno, axis=axis).copy()
        templateimage = np.take(fdata, msno, axis=axis)
        fedge, medge, fbin, mbin = rg.getContours(templateimage, dataimage)

        f,m = rg.get2DCorrespondences(fedge, medge, fbin, mbin)
        if(len(f) !=0 ):
            flist.append(np.hstack([np.zeros((len(f),1))+sno, f]))
            mlist.append(np.hstack([np.zeros((len(m),1))+sno , m]))

    fpoints = np.concatenate(flist)
    mpoints = np.concatenate(mlist)

    fIndices = fpoints[:,0]* ny*nz + fpoints[:,1]*nz +fpoints[:,2]
    fIndices = fIndices.astype(int)
    
    Ycount[fIndices] +=1
    Zcount[fIndices] +=1
    Yd[fIndices] += mpoints[:,1] - fpoints[:,1]
    Zd[fIndices] += mpoints[:,2] - fpoints[:,2]
    
    
    start = time.time()
    A = laplacianA3D(fdata.shape, Ycount.nonzero()[0])

    dy = lgmres(A, Yd , tol =1e-2)[0]
    print("dx calculated in {}s".format(time.time()- start))


    dz = lgmres(A, Zd, tol =1e-2)[0]
    print("dz calculated in {}s".format(time.time()- start))

    deformationField[0] = np.zeros(fdata.shape)
    deformationField[1] = dy.reshape(fdata.shape)
    deformationField[2] = dz.reshape(fdata.shape)
    
    return deformationField


def areaKeyFrame2DLaplacian(fixedImage, movingImage ,spacing, fthresh, mthresh, axis =0 ):

    fdata , mdata = loadNiiImages([fixedImage, movingImage], scale=True)


    nx, ny, nz  = fdata.shape
    nd  = len(fdata.shape)
    
    deformationField = np.zeros((nd, nx, ny, nz))
    rd = rg.RegistrationData(fdata, mdata, spacing, fthresh, mthresh)
    rd.scale = 1
    
    mpoints = rd.mpoints.copy()
    fpoints = rd.fpoints.copy()
    S = rg.getScaleMatrix(rd.fbinary, rd.mbinary)
    mpoints= rg.affineTransformPointCloud(rd.mpoints,S)
    T = rg.getCenterTranslationMatrix(fpoints, mpoints)
    mpoints= rg.affineTransformPointCloud(rd.mpoints,T@S)
    R = rg.align_principle_axes(fpoints, mpoints)
    R = np.eye(4)
    A1 = R@T@S
    #mpoints = affineTransformPointCloud(rd.mpoints,A1)
    rd.set_mpoints(mpoints)
    rd.estimate_fnormals()
    rd.estimate_mnormals()

    A2 = rg.icp_registration(rd)

    A= A2@A1
    A[3] = [0,0,0,1]
    rd.applyAffineTransform(A)
    
    fkeys, mkeys = rg.calculateKeyPoints(rd.fbinary, rd.mbinary)
    
    n = rd.fdata.shape[0]
    valuesx = np.zeros(n)
    valuesx[fkeys]=mkeys-fkeys
    A = laplacianA1D(n, fkeys)
    dx = lgmres(A, valuesx)[0]
    
    msno =[]
    for sno in range(rd.fdata.shape[axis]):
        tsno = int(sno + dx[sno])
        if(tsno<0):
            tsno=0
        if(tsno>=n):
            tsno = n-1
        msno.append(tsno)
    deformationField = sliceToSlice2DLaplacian(rd.fdata, rd.mdata, sliceMatchList = msno, axis = axis)

    transformedData   = rg.applyDeformationField(rd.mdata , deformationField)
    return A2@A1 , deformationField, transformedData

def areaKeyFrame3DLaplacian(fixedImage, movingImage , spacing, fthresh, mthresh, axis =0 ):

    fdata, mdata = loadNiiImages([fixedImage, movingImage], scale=True)

    nx, ny, nz  = fdata.shape
    nd  = len(fdata.shape)
    
    deformationField = np.zeros((nd, nx, ny, nz))
    rd = rg.RegistrationData(fdata, mdata, spacing, fthresh, mthresh)
    rd.scale = 1
    
    mpoints = rd.mpoints.copy()
    fpoints = rd.fpoints.copy()
    S = rg.getScaleMatrix(rd.fbinary, rd.mbinary)
    mpoints= rg.affineTransformPointCloud(rd.mpoints,S)
    T = rg.getCenterTranslationMatrix(fpoints, mpoints)
    mpoints= rg.affineTransformPointCloud(rd.mpoints,T@S)
    R = rg.align_principle_axes(fpoints, mpoints)
    R = np.eye(4)
    A1 = R@T@S
    #mpoints = affineTransformPointCloud(rd.mpoints,A1)
    rd.set_mpoints(mpoints)
    rd.estimate_fnormals()
    rd.estimate_mnormals()

    A2 = rg.icp_registration(rd)

    A= A2@A1
    A[3] = [0,0,0,1]
    rd.applyAffineTransform(A)
    
    fkeys, mkeys = rg.calculateKeyPoints(rd.fbinary, rd.mbinary)
    
    n = rd.fdata.shape[0]
    valuesx = np.zeros(n)
    valuesx[fkeys]=mkeys-fkeys
    A = laplacianA1D(n, fkeys)
    dx = lgmres(A, valuesx)[0]
    
    msno =[]
    for sno in range(rd.fdata.shape[axis]):
        tsno = int(sno + dx[sno])
        if(tsno<0):
            tsno=0
        if(tsno>=n):
            tsno = n-1
        msno.append(tsno)
    deformationField = sliceToSlice3DLaplacian(rd.fdata, rd.mdata, sliceMatchList = msno, axis = axis)
    transformedData   = rg.applyDeformationField(rd.mdata , deformationField)
    
    return A2@A1 , deformationField, transformedData


