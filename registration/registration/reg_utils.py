"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
Atchuth Naveen
Code developed at UC Irvine.

Implemented various functions that serve as building blocks in the registration pipeline
"""

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from functools import partial

import scipy
import skimage
from scipy.sparse.linalg import lgmres
from scipy.ndimage import affine_transform,geometric_transform
from scipy.ndimage import geometric_transform


from skimage import feature, measure

import open3d as o3d
import open3d
from registration.edge_utils import extract_edges, create_surface, create_surfacex
from registration.laplacianUtils import laplacianA2D
from registration.vol2affine import vol2affine
from registration.utils import loadNiiImages



def getAlignAxisAffineMatrix(fixed, moving):
    
    
    R, r1, r2 = vol2affine(moving=moving, template=fixed,pivot=(0, 0, 0))
    #R = align_rotation(r1,r2)
    origin = np.array(moving.shape)/2

    A1 = np.eye(4)
    A1[0:3, 3] = -origin

    A2 = np.eye(4)
    A2[0:3,0:3] = R[0:3,0:3]

    A3 = np.eye(4)
    A3[0:3, 3] = origin
    A = (A3@A2)@A1
    return A

def axisAlignData(fixedImage, movingImage):

    fdata, mdata = loadNiiImages([fixedImage, movingImage])

    A = getAlignAxisAffineMatrix(fdata, mdata)
    alignedData = affine_transform(mdata,np.linalg.inv(A), output_shape = mdata.shape, order =1)
    alignedData[alignedData<0] =0
    return A, alignedData

def getDataContours(dataImage):
    """
    Calculates internal and outer contours for data image.

    Parameters
    -----------
    dataImage : 2D image slice
    """
    dataImage[dataImage>500] = 500
    dataImage[dataImage<0] = 0

    data = skimage.exposure.equalize_adapthist(dataImage.astype(np.uint16))*255
    local_thresh = skimage.filters.threshold_otsu(data)
    binary = data>local_thresh

    edges = feature.canny(binary, sigma=3)
    all_labels = measure.label(edges)
    
    for label in range(np.max(all_labels)):
        if( np.sum(all_labels==label)<100):
            edges[all_labels==label] = 0

    edges = skimage.morphology.thin(edges)

    return edges, binary

def getTemplateContours(templateImage):
    """
    Calculates internal and outer contours for template image.

    Parameters
    -----------
    templateImage : 2D image slice
    """

    local_thresh = skimage.filters.threshold_otsu(templateImage)
    binary = templateImage>local_thresh
    edges = feature.canny(binary, sigma=3)
    edges = skimage.morphology.thin(edges)

    all_labels = measure.label(edges)

    for label in range(np.max(all_labels)):
        if( np.sum(all_labels==label)<25):
            edges[all_labels==label] = 0
    return edges, binary

def getContours(templateImage , dataImage):
    """
    Should generalise both the functions
    """
    fedge, fbin = getTemplateContours(templateImage)
    medge, mbin = getDataContours(dataImage)

    return fedge, medge, fbin, mbin
'''
def getContours(dataimage, templateimage):

    fedge , fbin = getDataContours(dataImage)
    medge , mbin = getTemplateContours(templateImage)

    return fedge, medge, fbin, mbin
'''
def get2DCorrespondences(fsection, msection, fbinary, mbinary , inner=True):
    """
    Matches points on contours along  
    
    """
    mpoints, mnormals = estimate2Dnormals(np.array(msection.nonzero()).T, mbinary)
    fpoints, fnormals = estimate2Dnormals(np.array(fsection.nonzero()).T, fbinary)

    fkdtree = scipy.spatial.KDTree(fpoints)
    mkdtree = scipy.spatial.KDTree(mpoints)
    #valuesx = np.zeros(fsection.shape[0]*fsection.shape[1])
    #valuesy = np.zeros(fsection.shape[0]*fsection.shape[1])

    if len(fpoints) <=0 or len(mpoints) <=0:
        return [],[]
    correspondences =[]
    for s,point in enumerate(fpoints):
        #point = mpoints_rcs[s]
        correspondences.append(get_correspondences( point, fnormals[s],mnormals, mkdtree, 5,30 ))
    correspondences = np.array(correspondences)

    c = correspondences!=-1
    cid = fpoints[c,0]* fsection.shape[1] + fpoints[c,1]

    if len(cid) <5:
        return [], []

    cindices = cid.astype(int)

    dx = mpoints[correspondences[c],0] - fpoints[c,0]
    dy = mpoints[correspondences[c],1] - fpoints[c,1]

    #print(np.mean(dx), np.percentile(dx,90), np.max(dx))

    valid_idx = abs(dx) < max(10,np.percentile(abs(dx), 90))
    valid_idy = abs(dy) < max(10,np.percentile(abs(dy) ,90))

    valid_id = np.array(valid_idx.astype(int)+valid_idy.astype(int)) ==2
    #print(np.sum(valid_id) , )
    f_ = fpoints[c]
    m_ = mpoints[correspondences[c]]
    return f_[valid_id] , m_[valid_id]


def shift2Dfunc(point, dx, dy):
    px = point[0] + dx[point[0], point[1]]
    py = point[1] + dy[point[0], point[1]]
    if(px<0 or px> dx.shape[0]):
        return (point[0], point[1])
    if(py<0 or py> dx.shape[1]):
        return (point[0], point[1])
    return (px, py)

def shift3Dfunc(point, dx, dy, dz):
    px = point[0] + dx[point[0], point[1],  point[2]]
    py = point[1] + dy[point[0], point[1],  point[2]]
    pz = point[2] + dz[point[0], point[1],  point[2]]
    if(px<0 or px> dx.shape[0]):
        return (point[0], point[1],point[2])
    if(py<0 or py> dx.shape[1]):
        return (point[0], point[1],point[2])
    if(pz<0 or pz> dx.shape[2]):
        return (point[0], point[1],point[2])
    return (px, py, pz)

def applyAffineTransform(image , A, output_shape):
    """
    Applies Affine Transform defined  by A on the numpy image data represented by image

    Parameters:
    -----------
    image: Can be Numpy array or a path to nii Image
    A : Affine matrix of 3x4
    """
    data = loadNiiImages([image])
    transformedData = affine_transform(data,np.linalg.inv(A), output_shape = output_shape)
    return transformedData

def applyDeformationField(image, deformationField):

    """
    Morphs the numpy iage data according to the deformation Field 

    Parameters:
    -----------
    image: Can be Numpy array or a path to nii Image
    """
    data = loadNiiImages([image])
    transformedData = geometric_transform(data, partial(shift3Dfunc, dx=deformationField[0], dy=deformationField[1], dz = deformationField[2]))
    return transformedData

def get_correspondences( sourcep,sourcen, targetn, kdtree, degree_thresh =5, distance_neighbours=500, dthresh = 5):
    if type(kdtree) == open3d.cpu.pybind.geometry.KDTreeFlann:
        [k, indices, _] = kdtree.search_knn_vector_3d(sourcep, distance_neighbours)
        target_npc = targetn[indices]
    elif type(kdtree) == scipy.spatial._kdtree.KDTree:
        d,indices =kdtree.query(sourcep, distance_neighbours)
        indices = indices[np.where(d<np.percentile(d,90))[0]]
        target_npc = targetn[indices]
    
    similarity = np.inner(target_npc,sourcen)
    degree_thresh = np.cos(degree_thresh*np.pi/180)
    try:
        idx = np.where(similarity >= degree_thresh)[0][0]
    except Exception as e:
        return -1
    return indices[idx]

def icp_registration(rd, itr = 25, tolerance =1e-2):
    
    def get_tolerance(source, X):
        source_ = source@X
        tol = np.mean(np.linalg.norm(source_-source , axis =1))
        return source_, tol
    
    affine= np.eye(4)

    source_ = rd.mpoints
    target  = rd.fpoints
    sourceN = rd.mnormals
    targetN = rd.fnormals

    source =  np.hstack([source_, np.ones((source_.shape[0],1))])

    tol_list=[0]
    for _ in tqdm(range(itr)):
        correspondences =[]
        sampl = np.random.randint(low=0, high=source.shape[0], size=10000)

        for s in sampl:
            correspondences.append(get_correspondences(source_[s], sourceN[s] , targetN,  rd.fkdtree))

        correspondences = np.array(correspondences)
        mset = source[sampl[correspondences!=-1]]
        fset = target[correspondences[correspondences!=-1]]

        A = mset
        B = np.hstack([fset, np.ones((fset.shape[0],1))])

        X = np.linalg.lstsq(A,B, rcond=None)[0]
        X[ :,3] = [0,0,0,1]

        source, tol = get_tolerance(source,X)
        tol_list.append(tol)
        #print(tol)
        affine= affine@X

        if(abs(tol_list[-1] - tol_list[-2])<tolerance):
            break

    #self.set_mpcd_points(source[:,0:3])
    #self.transforms.append(affine.T)
    return affine.T

def nonLinearRegistration2D(fsection, msection, fbinary, mbinary,  tol=1e-2 ):
    """
    Performs non linear laplacian registration between 2D images to align msection with fsection
    """

    def createDirchletBoundary2D(fsection, msection, fbinary, mbinary):
        """
        Helper function to create dirchlet boundary
        """
        fpoints, mpoints = get2DCorrespondences(fsection, msection, fbinary, mbinary)
        valuesx = np.zeros(fsection.shape[0]*fsection.shape[1])
        valuesy = np.zeros(fsection.shape[0]*fsection.shape[1])

        if len(fpoints) <=5 or len(mpoints) <=5:
            return valuesx, valuesy, []

        cindices = fpoints[:,0] * fsection.shape[1] + fpoints[:,1]
        cindices = cindices.astype(int)

        valuesx[cindices] = mpoints[:,0] - fpoints[:,0]
        valuesy[cindices] = mpoints[:,1] - fpoints[:,1]

        return valuesx, valuesy, cindices

    vx, vy, cid = createDirchletBoundary2D(fsection, msection, fbinary, mbinary)
    
    if len(cid) <5:
        dx = np.zeros(fsection.shape)
        return dx,dx
    A= laplacianA2D(fsection.shape, cid)
    dx = lgmres(A, vx.flatten() , tol =tol)[0]
    dy = lgmres(A, vy.flatten() , tol =tol)[0]
    dx = dx.reshape(fsection.shape)
    dy = dy.reshape(fsection.shape)
    return dx,dy

def affineTransformPointCloud(points, A):
    points_ = points.copy()
    points_ = A@np.hstack([points, np.ones((points_.shape[0],1))]).T
    return points_[:3].T

def estimate_normal(point, neighbours):
    """
    """
    centroid = np.mean(neighbours,axis=0)
    p_centered = neighbours -centroid
    point = point - centroid

    try:
        v = np.linalg.svd(p_centered - point)[-1]
        n =v[-1]
    except Exception:
        return None
    return n

def orient2Dnormals(points, normals, section):
    """
    Orients Normals to point away from center
    
    TODO: Make them point towards low intensity
    """
    section_ = section.flatten()

    #pIndex = points[:,0]*section.shape[1]+points[:,1]
    pFlatIndexMax = np.array(points[:,0]+9*normals[:,0]).astype(int)*section.shape[1]+np.array(points[:,1]+9*normals[:,1]).astype(int)

    points = points[np.where(pFlatIndexMax< len(section_))[0]]
    normals = normals[np.where(pFlatIndexMax< len(section_))[0]]

    pFlatIndexMax = np.array(points[:,0]+9*normals[:,0]).astype(int)*section.shape[1]+np.array(points[:,1]+9*normals[:,1]).astype(int)
    points = points[np.where(pFlatIndexMax> 0)[0]]
    normals = normals[np.where(pFlatIndexMax> 0)[0]]

    pFlatIndexMin = np.array(points[:,0]-9*normals[:,0]).astype(int)*section.shape[1]+np.array(points[:,1]-9*normals[:,1]).astype(int)

    points = points[np.where(pFlatIndexMin< len(section_))[0]]
    normals = normals[np.where(pFlatIndexMin< len(section_))[0]]
    pFlatIndexMin = np.array(points[:,0]-9*normals[:,0]).astype(int)*section.shape[1]+np.array(points[:,1]-9*normals[:,1]).astype(int)
    points = points[np.where(pFlatIndexMin> 0)[0]]
    normals = normals[np.where(pFlatIndexMin> 0)[0]]

    leftSum = np.zeros(normals.shape[0])
    rightSum = np.zeros(normals.shape[0])
    normalDirection = np.zeros(normals.shape[0])
    for k in range(1,10):
        leftSum  += section_[np.array(points[:,0]+k*normals[:,0]).astype(int)*section.shape[1]+np.array(points[:,1]+k*normals[:,1]).astype(int)]
        rightSum += section_[np.array(points[:,0]-k*normals[:,0]).astype(int)*section.shape[1]+np.array(points[:,1]-k*normals[:,1]).astype(int)]
        
    
    normalDirection[leftSum>=rightSum]=-1
    normalDirection[leftSum<rightSum] = 1

    normals = normals* np.expand_dims(normalDirection, axis=-1)

    return points, normals

def orient_normals_basic(points, normals):
    #Orients Normals to point away from center
     
    center = np.mean(points, axis=0)
    
    for i,point in enumerate(points):
        pp = point - center
        if(np.inner(pp,normals[i] )< 0):
            normals[i] = -normals[i]
    return points, normals

def orient_normals(points, normals, binary, k=9):
    """
    Orients Normals to make them point towards low intensity.
    For Each point, the sum of pixels in both directions of the normal vector are accumulated. 
    And the normal is made to point towards the sum containing low intensity.


    Parameters
    ----------
    points: Nx3 np array
    normals: Nx3 np array
    binary: 3D np array
    k: Number of pixels to consider on both sides. Default value is 9

    Returns:
    points: N_x3 np array. Some of the noisy points are filtered out
    normals: N_3 np array. Corresponding normal vectors with their sign flipped. 
    """

    binary_ = binary.flatten()
    sx, sy, sz = binary.shape

    #Filter Out points at the boundaries of the volume
    pFlatIndexMax = np.array(points[:,0]+k*normals[:,0]).astype(int)*sy*sz+np.array(points[:,1]+k*normals[:,1]).astype(int)*sz + np.array(points[:,1]+k*normals[:,2]).astype(int)
    points = points[np.where(pFlatIndexMax< len(binary_))[0]]
    normals = normals[np.where(pFlatIndexMax< len(binary_))[0]]

    pFlatIndexMax = np.array(points[:,0]+k*normals[:,0]).astype(int)*sy*sz+np.array(points[:,1]+k*normals[:,1]).astype(int)*sz + np.array(points[:,1]+k*normals[:,2]).astype(int)
    points = points[np.where(pFlatIndexMax> 0)[0]]
    normals = normals[np.where(pFlatIndexMax> 0)[0]]

    pFlatIndexMin = np.array(points[:,0]-k*normals[:,0]).astype(int)*sy*sz+np.array(points[:,1]-k*normals[:,1]).astype(int)*sz + np.array(points[:,1]-k*normals[:,2]).astype(int)
    points = points[np.where(pFlatIndexMin< len(binary_))[0]]
    normals = normals[np.where(pFlatIndexMin< len(binary_))[0]]

    pFlatIndexMin = np.array(points[:,0]-k*normals[:,0]).astype(int)*sy*sz+np.array(points[:,1]-k*normals[:,1]).astype(int)*sz + np.array(points[:,1]-k*normals[:,2]).astype(int)
    points = points[np.where(pFlatIndexMin> 0)[0]]
    normals = normals[np.where(pFlatIndexMin> 0)[0]]

    leftSum = np.zeros(normals.shape[0])
    rightSum = np.zeros(normals.shape[0])
    normalDirection = np.zeros(normals.shape[0])
    for n in range(1,k+1):
        leftSum  += binary_[np.array(points[:,0]+n*normals[:,0]).astype(int)*sy*sz+np.array(points[:,1]+n*normals[:,1]).astype(int)*sz + np.array(points[:,1]+n*normals[:,2]).astype(int) ]
        rightSum += binary_[np.array(points[:,0]-n*normals[:,0]).astype(int)*sy*sz+np.array(points[:,1]-n*normals[:,1]).astype(int)*sz + np.array(points[:,1]-n*normals[:,2]).astype(int)]
        
    
    normalDirection[leftSum>=rightSum]=-1
    normalDirection[leftSum<rightSum] = 1

    normals = normals* np.expand_dims(normalDirection, axis=-1)

    return points, normals


def estimate2Dnormals(points,binarySection=None , radius = 3,pkdtree = None,  progressbar= False):
    """
    points: Nx2 edge points from a 2d image
    Returns
    -----------
    points : N_ x2 valid points where normals are defined
    normals: N_ x2 normals at valid points
    """

    normals=np.zeros(points.shape)
    if not pkdtree:
        pkdtree = scipy.spatial.KDTree(points)
    count =0
    for i,point in tqdm(enumerate(points), disable = not progressbar):
        indices =pkdtree.query_ball_point(point, radius) 
        neighbours = points[indices]
        if(len(indices)>=4):
            n = estimate_normal(point, neighbours)
            if n is not None:
                normals[i] = n
            else:
                points[i,:] = [0,0]
        else:
            points[i,:] = [0,0]
    if binarySection is not None:
        points, normals = orient2Dnormals(points, normals, binarySection)
    return points, normals

def process3DImage( image3D , thresh =1 , **kwargs):
    """
    kwargs[mkernel] =5
    kwargs[gkernel] =5 

    """
    imbinary  = image3D.copy()

    imbinary[imbinary>=thresh] = 255
    imbinary[imbinary<thresh] = 0

    imedge = extract_edges(imbinary, **kwargs)
    imedge = imedge.astype(bool)

    #imsurface = clean_edges(imedge)

    
    imsurface = create_surface(imedge)

    imsurface = create_surfacex(imsurface)

    return imbinary, imedge, imsurface

def getTopNPeaks(signal, n):
    peaks, d = scipy.signal.find_peaks(signal , prominence =1 )
    prominences = d["prominences"]
    return peaks[np.argsort(prominences)[-n:]]

def calculateKeyPoints(fbinary, mbinary):
    mpixelcount=[]
    for section in mbinary:
        mpixelcount.append(len(section.nonzero()[0]))
    msignal = np.max(mpixelcount) - mpixelcount
    m1, m2 = getTopNPeaks(msignal,2)
    m3, m4, m5 = getTopNPeaks(mpixelcount,3)
    mkeys = np.array([m1, m2, m3, m4, m5])
    fpixelcount=[]
    for section in fbinary:
        fpixelcount.append(len(section.nonzero()[0]))
    fsignal = np.max(fpixelcount) - fpixelcount
    f1, f2 = getTopNPeaks(fsignal,2)
    f3,f4,f5 = getTopNPeaks(fpixelcount,3)
    fkeys = np.array([f1, f2, f3, f4, f5])
    
    fidx = np.argsort(fkeys).astype(int)
    fkeys = fkeys[fidx]
    mkeys = mkeys[fidx]
    return fkeys, mkeys



def estimate3Dnormals(points,pkdtree,binary, radius = 3, method ="Center"):
    """
    method: "Center" - Normals are oriented to the center of the image,
            "Threshold" - Normals are oriented towards low intensity
    """
    normals=np.zeros(points.shape)
    for i,point in tqdm(enumerate(points)):
        [k, indices, _]  =pkdtree.search_radius_vector_3d(point, radius) 
        neighbours = points[np.array(indices)]
        if(k>6):
            n = estimate_normal(point, neighbours)
            if n is not None:
                normals[i] = n
            else:
                points[i,:] = [0,0,0]
        else:
            points[i,:] = [0,0,0]
    if method=="Center":
        points, normals = orient_normals_basic(points, normals)
    else:
        points, normals = orient_normals(points, normals, binary)
    return points, normals


def get_bounding_box(image3D, thresh= 2, display =False):
    
    pl_listx=[]
    pl_listy=[]
    pl_listz=[]
    
    for x in range (image3D.shape[0]):
        pl_listx.append(np.mean(image3D[x]))
    
    for y in range( image3D.shape[1]):
        pl_listy.append(np.mean(image3D[:,y,:]))
        
    for z in range( image3D.shape[2]):
        pl_listz.append(np.mean(image3D[:,:,z]))
    
    if display:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

        ax[0].plot(pl_listx)
        ax[0].set_title('x-axis', fontsize=20)

        ax[1].plot(pl_listy)
        ax[1].set_title(r'y-axis', fontsize=20)

        ax[2].plot(pl_listz)
        ax[2].set_title(r'z-axis', fontsize=20)

        fig.tight_layout()
        plt.show()
        
    px = np.where(np.array(pl_listx)>thresh)[0]
    py = np.where(np.array(pl_listy)>thresh)[0]
    pz = np.where(np.array(pl_listz)>thresh)[0]
    
    sx = px[-1]-px[0]+1
    sy = py[-1]-py[0]+1
    sz = pz[-1]-pz[0]+1

    return sx, sy, sz

def getScaleMatrix(fixedImage, movingImage, thresh=2):
    fx, fy, fz = get_bounding_box(fixedImage)
    mx, my, mz = get_bounding_box(movingImage)

    S = np.eye(4)
    S[0,0] = fx/mx
    S[1,1] = fy/my
    S[2,2] = fz/mz
    return S

def scalePointCloud(points, S):
    origin = np.mean(points, axis=0)
    A1 = np.eye(4)
    A1[0:3, 3] = -origin

    A2 = S

    A3 = np.eye(4)
    A3[0:3, 3] = origin
    A = (A3@A2)@A1
    return S, affineTransformPointCloud(points, A)

def getCenterTranslationMatrix(fpoints, mpoints):
    """
    Returns the translation Matrix that transforms the center of mpoints to fpoints.

    Parameters
    ----------
    fpoints: NxK numpy array of N K-dimension points
    mpoints: N'xK numpy array of N' K-dimension points
    """
    mcenter = np.mean(mpoints, axis=0)
    fcenter = np.mean(fpoints, axis=0)
    #mpoints  = mpoints + fcenter -mcenter
    T= np.eye(4)
    T[0:3, 3] = fcenter-mcenter
    return T

def align_centers(fpoints, mpoints):
    """
    Aligns the center of mpoints with fpoints.

    Parameters
    ----------
    fpoints: NxK numpy array of N K-dimension points
    mpoints: N'xK numpy array of N' K-dimension points
    """
    mcenter = np.mean(mpoints, axis=0)
    fcenter = np.mean(fpoints, axis=0)
    mpoints  = mpoints + fcenter -mcenter
    T= np.eye(4)
    T[0:3, 3] = fcenter-mcenter
    return mpoints, T

def get_sortedEigenVectors(points):
    
    center = np.mean(points, axis=0)
    points = points-center

    cov =np.cov(points.T)
    eig, eigv = np.linalg.eig(cov)
    idx = eig.argsort()[::-1]
    eig = eig[idx]
    eigv = eigv[:,idx]
    
    return eig,eigv
    
def get_pca_rot_matrix(fpoints, mpoints):
    """
    column is eigen vector
    """
    _, fv = get_sortedEigenVectors(fpoints)
    _, mv = get_sortedEigenVectors(mpoints)

    #Align axes direction to point in same direction, the axes can be flipped in some cases

    for i,v in enumerate(np.sum(fv*mv,axis = 0)<0):
        if v:
            mv[:,i] = -mv[:, i]

    R  = fv@np.linalg.inv(mv)
    return R

def align_principle_axes(fpoints, mpoints):
    
    R = get_pca_rot_matrix(fpoints, mpoints)

    origin = np.mean(mpoints, axis=0)
    
    #If principle components calculated with center as origin
    A1 = np.eye(4)
    A1[0:3, 3] = -origin

    A2 = np.eye(4)
    A2[0:3,0:3] = R

    A3 = np.eye(4)
    A3[0:3, 3] = origin
    A = (A3@A2)@A1

    return A

class RegistrationData(object):
    """
    Registration assumes fdata will be aligned to the shape of mdata
    """
    def __init__(self, fdata, mdata, spacing , fthresh=1, mthresh=100):
        
        self.fdata = fdata.copy()
        self.mdata = mdata.copy()
        
        self.mthresh = mthresh
        self.fthresh = fthresh
        
        self.mpoints = None
        self.fpoints = None
        self.mnormals = None
        self.fnormals = None
        
        self.spacing = spacing
        if(spacing == 100):
            self.mkernel= 3
            self.gkernel= 5
        if spacing ==50:
            self.mkernel =5
            self.gkernel =5
        if spacing ==25:
            self.mkernel =10
            self.gkernel =10
        if spacing ==10:
            self.mkernel =20
            self.gkernel =15

        
        self.mpcd = o3d.geometry.PointCloud()
        self.fpcd =  o3d.geometry.PointCloud()
        
        self.mkdtree = None
        self.fkdtree = None
        
        self.createPointClouds()

    def createPointCloud(self, imsurface):

        return np.asarray(imsurface.nonzero()).T

    def processFixedImage(self):
        imbinary, imedge, imsurface = process3DImage(self.fdata, self.fthresh,  mkernel = self.mkernel, gkernel = self.gkernel)
        self.fbinary = imbinary
        self.fedge = imedge
        self.fsurface = imsurface

    def processMovingImage(self):
        imbinary, imedge, imsurface = process3DImage(self.mdata, self.mthresh,  mkernel = self.mkernel, gkernel = self.gkernel)
        self.mbinary = imbinary
        self.medge = imedge
        self.msurface = imsurface


    def createPointClouds(self):

        self.processMovingImage()
        self.processFixedImage()
        fpoints = self.createPointCloud(self.fsurface)
        mpoints = self.createPointCloud(self.msurface)

        self.set_mpoints(mpoints)
        self.set_fpoints(fpoints)  

    def estimate_fnormals(self):
        points, normals  = estimate3Dnormals(self.fpoints, self.fkdtree, self.fbinary)
        self.set_fnormals(normals)
        self.set_fpoints(points)

    def estimate_mnormals(self):
        points, normals  = estimate3Dnormals(self.mpoints, self.mkdtree, self.mbinary)
        self.set_mnormals(normals)
        self.set_mpoints(points)


    def set_mpcd_points(self, points):
        self.mpcd.points = o3d.utility.Vector3dVector(points)
        self.mkdtree =  o3d.geometry.KDTreeFlann(self.mpcd)
        
    def set_fpcd_points(self, points):
        self.fpcd.points = o3d.utility.Vector3dVector(points)
        self.fkdtree =  o3d.geometry.KDTreeFlann(self.fpcd)
        
    def applyAffineTransform(self,A):
        modifiedData = affine_transform(self.mdata,np.linalg.inv(A), output_shape = self.fdata.shape)
        self.mdata =modifiedData
        self.processMovingImage()
        mpoints  = self.createPointCloud(self.msurface)
        self.set_mpoints(mpoints)
        self.estimate_mnormals()
    def applyGeometricTransform(self, dx, dy, dz):
        def shift_func(point, dx, dy, dz):
            px = point[0] + dx[point[0], point[1],  point[2]]
            py = point[1] + dy[point[0], point[1],  point[2]]
            pz = point[2] + dz[point[0], point[1],  point[2]]

            if(px<0 or px> dx.shape[0]):
                return (point[0], point[1],point[2])
            if(py<0 or py> dx.shape[0]):
                return (point[0], point[1],point[2])
            if(pz<0 or pz> dx.shape[0]):
                return (point[0], point[1],point[2])
            return (px, py, pz)
        
        self.fdata = geometric_transform(self.fdata, partial(shift_func, dx=dx, dy=dy, dz=dz), order =0)

        self.processFixedImage()
        fpoints  = self.createPointCloud(self.fsurface)
        self.set_fpoints(fpoints)
  
    def set_mpoints(self, points):
        self.mpoints = points
        self.mpcd.points = o3d.utility.Vector3dVector(points)
        self.mkdtree =  o3d.geometry.KDTreeFlann(self.mpcd)
        
    def set_fpoints(self,points):
        self.fpoints = points
        self.fpcd.points = o3d.utility.Vector3dVector(points)
        self.fkdtree =  o3d.geometry.KDTreeFlann(self.fpcd)
        
    def set_mnormals(self,normals):
        self.mnormals = normals
        self.mpcd.normals = o3d.utility.Vector3dVector(normals)
        
    def set_fnormals(self,normals):
        self.fnormals = normals
        self.fpcd.normals = o3d.utility.Vector3dVector(normals)
        
    def get_mpoints(self):
        return self.mpoints
    
    def get_fpoints(self):
        return self.fpoints
    
    def get_mnormals(self):
        return self.mnormals
    
    def get_fnormals(self):
        return self.fnormals