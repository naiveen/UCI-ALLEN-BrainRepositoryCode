# Code for UCI ALLEN Brain Repository


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
