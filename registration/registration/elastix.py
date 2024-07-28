import glob
import numpy as np
import os
import subprocess

from registration.utils import loadNiiImages, create_nifti_image
ELASTIXDIR = os.environ.get("ELASTIX_HOME")

def elastixRegistration(fixedImagePath, movingImagePath, outputDir, rescale=True):
    """
    Wrapper to run elastix registration from command line. Requires ELASTIXDIR to be defined as a global variable and expects the
    parameter file to present in current directory. 
    
    Parameters
    -------------
    fixedImagePath :  path to fixedImage
    movingImagePath : path to movingImage
    outputDir : path to outputDir

    Returns
    ------------
    path to deformed moving Image output
    """
    global ELASTIXDIR
    def rescaleMaxTo255(data):
        """
        Thresholds  and converts data to UINT8
        """
        maxVal  = np.percentile(data, 99)
        data[data > maxVal ] = maxVal
        data = data*255 /maxVal
        return data


    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    if rescale:
        mdata = loadNiiImages([movingImagePath])
        rescaledData = rescaleMaxTo255(mdata)
        rescaledDataPath = os.path.join(outputDir, "rescaled.nii.gz")
        create_nifti_image(rescaledData,2.5,rescaledDataPath,1)
        movingImagePath = rescaledDataPath

    assert ELASTIXDIR is not None, "ELASTIXDIR not defined"

    registration_cmd = [ELASTIXDIR + '/elastix',"-f",fixedImagePath, "-m", movingImagePath, "-out", outputDir, "-p" , "001_parameters_Rigid.txt", "-p", "002_parameters_BSpline.txt"]
    subprocess.run(registration_cmd)
    return os.path.join(outputDir, "result.1.nii")

def elastixTransformation(imagePath, regDir, outDir=None):
    """
    Wrapper to run elastix transform command line to apply transformation to the given image path based on elastix registration. 
    If outPath is not given, it is stored in regDir/transform.nii
    Parameters:
    """
    global ELASTIXDIR
    if outDir is None:
        outDir = regDir

    if not os.path.isdir(outDir):
        os.mkdir(outDir)
        
    outPath = os.path.join(outDir, "result.nii")

    assert ELASTIXDIR is not None, "ELASTIXDIR not defined"
    transformix_cmd = [ELASTIXDIR + "/transformix","-in",imagePath,"-out", outDir,"-tp",os.path.join(regDir,"TransformParameters.1.txt")]
    subprocess.run(transformix_cmd)
    return outPath
