import scipy.io
import numpy as np
import subprocess
import argparse 
import os
import nibabel as nib
from collections import Counter
import pandas as pd
from cell_regions import *

def get_input_cell_locations(input_mat):
    mat = scipy.io.loadmat(input_mat)
    location_arr  = mat['location'][0]
    prediction_arr = mat['predictions'][0]
    cell_locations = []
    for i in range(location_arr.shape[0]):
        location = location_arr[i]
        prediction = prediction_arr[i]
        location = np.squeeze(location)
        prediction = np.squeeze(prediction)
        if(len(location) ==0):
            continue
        true_locations = location.T[prediction!=2]
        print(true_locations.shape)
        true_locations = np.hstack((true_locations, np.zeros((true_locations.shape[0],1))+i))
        cell_locations.append(true_locations)
    cell_locations = np.vstack(cell_locations) 
    return cell_locations



def get_registered_points(file):
    output_points=[]
    with open(file, "r") as f:
        for line in f:
            line = line.split(";")
            output_point  = [float(x) for x in line[4].split()[3:6]]
            output_points.append(output_point)
    output_points= np.asarray(output_points)
    return output_points

def get_registered_regions(output_points, ann_data, inp):
    region_ids =[]
    exterior_points =0
    for i,point in enumerate(output_points):
        try:

            point = np.asarray(point).astype(int)
            if(np.sum(point<0)!=0):
                continue
            if(point[1] >= ann_data.shape[1] or point[2] >= ann_data.shape[2] or point[0]>= ann_data.shape[0]):
                exterior_points = exterior_points+1
                continue
            region_ids.append(int(ann_data[point[0], point[1], point[2]]))
        except Exception as e:
            print(i, point, inp[i])
    cell_counts = Counter(region_ids)
    print("Exterior Points :{}".format(exterior_points))
    return cell_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--fixed_image', type=str) #.\reference\average_template_25.nii
    parser.add_argument('--moving_image', type=str) #output\brain25\result.1.nii
    parser.add_argument('--input_mat', type=str, default = None) #'AI_result.mat'
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    fixed_image = args.fixed_image
    moving_image = args.moving_image
    output_dir = args.output_dir
    input_mat = args.input_mat

    input_points_file = os.path.join(args.output_dir, "inputpoints.txt")
    output_points_file = os.path.join(args.output_dir, "output_points.txt")
    cell_count_file = os.path.join(args.output_dir, "cell_count.csv")
    
    registration_cmd = ["./elastix/elastix","-m",fixed_image, "-f", moving_image, "-out", output_dir, "-p" , "001_parameters_Rigid.txt", "-p", "002_parameters_BSpline.txt"]
    transformix_cmd  = [ "./elastix/transformix","-def",input_points_file,"-out", output_dir,"-tp",os.path.join(output_dir,"TransformParameters.1.txt")]
    transformix_cmd2  = [ "./elastix/transformix","-in","CCF_DATA/annotation_25.nii","-out", output_dir,"-tp",os.path.join(output_dir,"TransformParameters.1.txt")]




    #Adjust Scale
    mImage = nib.load(moving_image)
    mData = mImage.get_fdata()
    #subprocess.run(registration_cmd) 

    if args.input_mat:
        cell_locations =get_input_cell_locations(input_mat)
        cell_locations = cell_locations[:, [2,1,0]]
        cell_locations = np.round(cell_locations*[ 1, 1/20,1/20]).astype(int)
        cell_locations[:,0] = mData.shape[0] - cell_locations[:,0]-1
        cell_locations[:,1] = mData.shape[1] - cell_locations[:,1]-1
        cell_locations[:,2] = mData.shape[2] - cell_locations[:,2]-1



        np.savetxt(input_points_file, cell_locations , "%d %d %d", header = "point\n"+str(cell_locations.shape[0]), comments ="")
        
        #subprocess.run(transformix_cmd)
        subprocess.run(transformix_cmd2)


        output_points = get_registered_points(os.path.join(output_dir,"outputpoints.txt"))
        

        np.savetxt(output_points_file, output_points , "%d %d %d", header = "point\n"+str(cell_locations.shape[0]), comments ="")

        print(len(cell_locations))


        annotation_image  = nib.load(os.path.join(output_dir,"result.nii"))
        ann_data = annotation_image.get_fdata()

        #cell_region_counts  = get_registered_regions(output_points, ann_data, cell_locations)

        cell_region_counts  = get_registered_regions(cell_locations, ann_data, cell_locations)
        pd.DataFrame(dict(cell_region_counts).items(), columns=["region", "count"]).to_csv(cell_count_file, index=False)

        atlas_df = pd.read_csv(r"G:/Brain_Stitch/CCF_DATA/1_adult_mouse_brain_graph_mapping.csv", index_col=None)
        count_df = pd.read_csv(cell_count_file, index_col=None)


        region_df,count_df = process_counts(atlas_df, count_df)
        count_df.to_csv(os.path.join(output_dir,"cell_region_count.csv"), index=False)
        region_df.to_csv(os.path.join(output_dir,"region_counts.csv"), index=False)

