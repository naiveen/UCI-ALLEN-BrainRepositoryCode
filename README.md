# Code for UCI ALLEN Brain Repository


## Registration and Cell Counting

* Registration currently requires Elastix to be placed in the folder in which the script will be run. Elastix binaries can be downloaded at https://github.com/SuperElastix/elastix/releases/tag/5.1.0

* This script can generate nii files and perform registration as well as use existing nii files for registration. 
    
* If --img_dir is provided, the script creates nii files according to the channel specified and uses them for registration.

* If --img_dir is not provided, then script takes additional arguments --fixed_image, --moving_image. 

* In addition output_dir can be specified to store all intermediary and final results. 

* For performing cell detection and cell counting, additional arguments '--cell_detection' need to be given. For cell detection, '--img_dir' is a required argument. 
    A threshold can be adjusted for cell counting with the argument '--threshold'

* --input_mat is only present for legacy reasons and can be ignored. 

* --t2d is to specify that the moving image is the template in registration. In registration generally, the template is fixed and the image data will be moving to align with the template.
**Note**: Currently cell counting is only supported with --t2d option and the other way around will be soon supported. 


* For help 
```
python run_registration_cellcounting.py -h
```

* Registration only command 
```
 python .\run_registration_cellcounting.py --fixed_image 'CCF_DATA/average_template_25.nii.gz' --moving_image '../registration/B39/brain_25.nii.gz' --output_dir reg
```

* Registration and cell counting.
```
python .\run_registration_cellcounting.py --img_dir  IMGDIR --channel 0 --cell_detection --threshold 10

```



## To Convert the directory of tif sections into Zarr directory. 

img_dir - Input directory containing tif file

out_dir - Output directory to store Zarr chunks

--channel - optional argument to select channel

```bash
python tif_to_ome.py [img_dir] [out_dir] --channel 0 
```

## To visualize Zarr files on neuroglancer

```bash
git clone https://github.com/google/neuroglancer.git

# From the directory containing Zarr files

python neuroglancer/cors_webserver.py
```

Load data from the live demo at [https://neuroglancer-demo.appspot.com](https://neuroglancer-demo.appspot.com).
