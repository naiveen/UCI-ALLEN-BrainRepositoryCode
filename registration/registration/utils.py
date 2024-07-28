import os
import nibabel as nib
import numpy as np
import scipy.ndimage
import SimpleITK as sitk


def create_nifti_image(img_array, scale, name=None, sz= None):
    """
    img_array : numpy array, containing stack of images
    scale: nifti scale
    """


    # The following parameters are set to be consistent with ALLEN CCF NII Templates
    affine_transform = np.zeros((4,4))
    affine_transform[0,2] = 0.01 * scale
    affine_transform[2,1] = -0.01* scale
    if sz==None:
        affine_transform[1,0] = -0.05
    else:
        affine_transform[1,0] = -0.05 *sz
    affine_transform[3,3] = 1
    nibImg = nib.Nifti1Image(img_array,affine_transform)
    nibImg.header['qform_code'] = 1
    nibImg.header['qoffset_x'] = -5.695
    nibImg.header['qoffset_y'] = 5.35
    nibImg.header['qoffset_z'] = 5.22


    if name != None:
        if name[-1]!='z':
            name  = os.path.join(name, 'brain_{}.nii.gz'.format(int(scale*10))) 
        nibImg.to_filename( name)
    return nibImg

def loadNiiImages(imageList, scale = False):
    """
    imageList can contaiin both paths to .nii images or loaded nii images
    loads nii images from the paths provided in imageList and returns a list of 3D numpy array representing image data. 
    If numpy data is present in imageList, the same will be returned 
    """
    if scale:
        if type(imageList[0])==str:
            fImage = nib.load(imageList[0])
        else:
            scale = False

    images =[]
    for image in imageList:
        if type(image) == str:
            niiImage = nib.load(image)
            imdata = niiImage.get_fdata()

            # Execution is faster on copied data
            if scale:
                scales = tuple(np.array(niiImage.header.get_zooms()) / np.array(fImage.header.get_zooms()))

                imdata =  scipy.ndimage.zoom(imdata.copy(), scales, order=1)
            images.append(imdata.copy())

        else:
            images.append(image.copy()) 

    if (len(imageList)==1):
        return images[0]
    return images

def getMutualInformation(fdata, mdata):
    """
    Wrapper function to calculate Mutual information between two numpy arrays usint SITK mutual information
    
    Parameters
    ------------
    fdata: np array
    mdata: np array
    """
    def rescaleMaxTo255(data):
        """
        Thresholds  and converts data to UINT8
        """
        maxVal  = np.percentile(data, 99)
        data[data > maxVal ] = maxVal
        data = data*255 /maxVal
        return data

    mdata[mdata<0] =0
    m8 = rescaleMaxTo255(mdata)
    f8 = rescaleMaxTo255(fdata)
    fImage = sitk.GetImageFromArray(f8.astype(float))
    mImage = sitk.GetImageFromArray(m8.astype(float))
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation()
    return registration_method.MetricEvaluate(fImage, mImage)

