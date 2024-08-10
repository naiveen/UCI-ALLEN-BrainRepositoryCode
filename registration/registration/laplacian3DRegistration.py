"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
Atchuth Naveen
Code developed at UC Irvine.


Contains registration pipeline for performing non linear laplacian registration in 3D space.
"""

import numpy as np

from scipy.sparse.linalg import lgmres
from registration.utils import loadNiiImages
from registration.reg_utils import RegistrationData, getScaleMatrix, affineTransformPointCloud, getCenterTranslationMatrix, align_principle_axes, icp_registration, get_correspondences, applyDeformationField
from registration.laplacianUtils import laplacianA3D

from time import time
from tqdm import tqdm


def affineRegistration(fixedImagePath, movingImagePath, spacing, fthresh, mthresh, tol=1e-2):
    fdata, mdata = loadNiiImages([fixedImagePath, movingImagePath] , scale =True)
    rd = RegistrationData(fdata, mdata, spacing, fthresh, mthresh)
    rd.fnormals = None
    rd.mnormals = None
    rd.fbinary = None
    rd.mbinary = None
    rd.mkdtree = None
    rd.scale = None

    fpoints = rd.fpoints.copy()
    S = getScaleMatrix(rd.fbinary, rd.mbinary)
    mpoints= affineTransformPointCloud(rd.mpoints,S)
    T = getCenterTranslationMatrix(fpoints, mpoints)
    mpoints= affineTransformPointCloud(rd.mpoints,T@S)
    R = align_principle_axes(fpoints, mpoints)
    R = np.eye(4)
    A1 = R@T@S
    rd.set_mpoints(mpoints)
    rd.estimate_fnormals()
    rd.estimate_mnormals()

    A2 = icp_registration(rd)

    A= A2@A1
    A[3] = [0,0,0,1]
    rd.applyAffineTransform(A)

def reg3D(fixedImage, movingImage, spacing, fthresh, mthresh, tol=1e-2):

    
    fdata, mdata = loadNiiImages([fixedImage, movingImage] , scale =True)
    
    nx, ny, nz  = fdata.shape
    nd  = len(fdata.shape)
    
    deformationField = np.zeros((nd, nx, ny, nz))
    rd = RegistrationData(fdata, mdata, spacing, fthresh, mthresh)
    rd.scale = 1

    mpoints = rd.mpoints.copy()
    fpoints = rd.fpoints.copy()
    S = getScaleMatrix(rd.fbinary, rd.mbinary)
    mpoints= affineTransformPointCloud(rd.mpoints,S)
    T = getCenterTranslationMatrix(fpoints, mpoints)
    mpoints= affineTransformPointCloud(rd.mpoints,T@S)
    R = align_principle_axes(fpoints, mpoints)
    R = np.eye(4)
    A1 = R@T@S
    #mpoints = affineTransformPointCloud(rd.mpoints,A1)
    rd.set_mpoints(mpoints)
    rd.estimate_fnormals()
    rd.estimate_mnormals()

    A2 = icp_registration(rd)

    A= A2@A1
    A[3] = [0,0,0,1]
    rd.applyAffineTransform(A)
    dx, dy,dz = nonLinearRegistration(rd , tol =tol)

    deformationField[0] = dx
    deformationField[1] = dy
    deformationField[2] = dz
    transformedData   = applyDeformationField(rd.mdata , deformationField)
    return A,deformationField, transformedData

def nonLinearRegistration(rd, x0=None, y0=None, z0 = None, tol =1e-2):
    def createDirchletBoundary(rd):
        """

        Finds correspondences from moving data to fixed data
        """

        mpoints = rd.mpoints
        fpoints = rd.fpoints
        
        mnormals = rd.mnormals
        fnormals = rd.fnormals
        
        fd = rd.fdata
        correspondences =[]
        for s,point in tqdm(enumerate(fpoints)):
            #point = mpoints_rcs[s]
            correspondences.append(get_correspondences( point, fnormals[s], mnormals, rd.mkdtree, 5,100 ))
        correspondences = np.array(correspondences)
        c = correspondences!=-1
        cid = fpoints[c,0]* fd.shape[1]*fd.shape[2] + fpoints[c,1]*fd.shape[2] +fpoints[c,2]

        valuesx = np.zeros(fd.shape[0]*fd.shape[1]*fd.shape[2])
        valuesx[cid.astype(int)]=mpoints[correspondences[c],0] - fpoints[c,0]

        valuesy = np.zeros(fd.shape[0]*fd.shape[1]*fd.shape[2])
        valuesy[cid.astype(int)]=mpoints[correspondences[c],1] - fpoints[c,1]

        valuesz = np.zeros(fd.shape[0]*fd.shape[1]*fd.shape[2])
        valuesz[cid.astype(int)]=mpoints[correspondences[c],2] - fpoints[c,2]
        
        return valuesx, valuesy, valuesz, cid

    vx, vy, vz, cid = createDirchletBoundary(rd)
        
    A= laplacianA3D(rd.fdata.shape, cid)
    """
    import pyamg

    resx, solver = pyamg.solve(A,valuesx , return_solver = True)
    resy = pyamg.solve(A, valuesy, existing_solver= solver)
    resz = pyamg.solve(A, valuesz, existing_solver = solver)
    """
    start = time.time()
    if x0 is None:
        dx = lgmres(A, vx.flatten(), tol =tol)[0]
    else:
        dx = lgmres(A, vx.flatten() , x0 = x0.flatten(), tol =tol)[0]
    print("dx calculated in {}s".format(time.time()- start))
    if y0 is None:
        dy = lgmres(A, vy.flatten(), tol =tol)[0]
    else:
        dy = lgmres(A, vy.flatten(), x0 = y0.flatten(), tol =tol)[0]
    print("dy calculated in {}s".format(time.time()- start))
    if z0 is None:
        dz = lgmres(A, vz.flatten(), tol =tol)[0]
    else:
        dz = lgmres(A, vz.flatten(), x0 = z0.flatten(), tol =tol)[0]
    print("dz calculated in {}s".format(time.time()- start))
    
    dx = dx.reshape(rd.fdata.shape)
    dy = dy.reshape(rd.fdata.shape)
    dz = dz.reshape(rd.fdata.shape)
    
    return dx,dy,dz
