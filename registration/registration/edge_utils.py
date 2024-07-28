import matplotlib.pyplot as plt
import numpy as np
import skimage

from scipy.ndimage import gaussian_filter
from skimage import feature, filters, measure   # type: ignore
from skimage.util import random_noise
from tqdm import tqdm
from joblib import Parallel, delayed


def extract_edge2d(image2d, thresh =100, mkernel =5, gkernel =5):
    image  = filters.median(image2d, np.ones((mkernel, mkernel)))
    image = gaussian_filter(image, sigma=2, radius=gkernel)
    edges1 = feature.canny(image, sigma=1)
    edges2 = feature.canny(image, sigma=2)
    edges3 = feature.canny(image, sigma=3)

    edges2[edges3] =True
    all_labels = measure.label(edges2)
    for label in range(np.max(all_labels)):
        if( np.sum(all_labels==label)<thresh):
            edges2[all_labels==label] = False

    edges1[edges2] =True
    all_labels = measure.label(edges1)
    for label in range(np.max(all_labels)):
        if( np.sum(all_labels==label)<thresh):
            edges1[all_labels==label] = 0

    return edges1



def extract_edges(image3d , sigma=3, thresh =100, mkernel =5, gkernel=5):
    
    def extract_edge2d_pwrapper(index, image2d, thresh, mkernel, gkernel):
        edge2d = extract_edge2d(image2d, thresh, mkernel, gkernel)
        return index, edge2d

    edge3d =  np.zeros(image3d.shape)
    results = Parallel(n_jobs=-4)(delayed(extract_edge2d_pwrapper)(i, image2d, thresh = thresh, mkernel =mkernel, gkernel=gkernel) for i,image2d in enumerate(image3d))
    if not results:
        raise ValueError("Edge detection failed")
    for i, edge2d in results:
        edge3d[i]= edge2d
    """
    for i in range(image3d.shape[0]):
        image2d = image3d[i]
        edge2d = extract_edge2d(image2d,100)
        edge3d[i] = edge2d
    """
    return edge3d

def clean_edges(image, thresh =6):
    all_labels = measure.label(image)
    label_count  =[]
    for label in range(1, np.max(all_labels)+1):
        label_count.append([np.sum(all_labels==label) , label])
    
    label_count = sorted(label_count, reverse = True)
    if(len(label_count)<thresh):
        return image
    for _, label in label_count[thresh:]:
        image[all_labels==label] = False
    return image


def create_surface(image3D, thresh =3):
    surface = image3D.copy()
    for i in range(image3D.shape[0]):
        outer2D = skimage.morphology.thin(image3D[i]) # type: ignore
        surface[i] = clean_edges(outer2D, 3)
    return surface

def create_surfacex(edge3d: np.ndarray) -> np.ndarray:
    assert edge3d.ndim == 3
    X,Y,Z = edge3d.shape
    surface = np.zeros(edge3d.shape)
    for x in range(X):
        for y in range(Y):
            row  = edge3d[x,y]
            rnz  = row.nonzero()[0]
            if rnz.size!=0:
                z1 =rnz[0]
                z2 =rnz[-1]
                surface[x,y,z1] =1
                surface[x,y,z2] =1

    for z in range(Z):
        for x in range(X):
            row  = edge3d[x,:,z]
            rnz  = row.nonzero()[0]
            if rnz.size!=0:
                y1 =rnz[0]
                y2 =rnz[-1]
                surface[x,y1,z] =1
                surface[x,y2,z] =1
    return surface