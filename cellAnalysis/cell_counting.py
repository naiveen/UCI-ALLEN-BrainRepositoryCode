import pandas as pd
import argparse
import os

def arg_parser():
    parser = argparse.ArgumentParser(description='Refine cell counts')
    parser.add_argument('--csv', type=str,
                        help='path to csv file')
    parser.add_argument('--out', type=str, default="",
                        help='output_dir')
    parser.add_argument('--level', type=int, default=5,
                        help='number of levels to be included in final counts')
    return parser


def process_counts(atlas_df, count_df, nlevel =11):
	

	count_df["region"] =count_df["region"].map(lambda x: int(x))

	count_df["region name"] =  count_df["region"].map(lambda x: atlas_df[atlas_df["parcellation_index"] ==x]["name"].values)

	count_df["region name"] = count_df["region name"].map(lambda x : x[0] if len(x)>0 else "" ) 
	

	count_df["structure_id_path"] = count_df["region"].map(lambda x: atlas_df[atlas_df["parcellation_index"] ==x]["structure_id_path"].values)

	count_df["structure_id_path"] = count_df["structure_id_path"].map(lambda x : x[0] if len(x)>0 else "")
	count_df["structure_ids"] =  count_df["structure_id_path"].map(lambda x : x.split("/"))

	
	

	for i in range(11):
	    level = "level_{}".format(i)
	    count_df[level] =  count_df["structure_ids"].map(lambda x : int(x[i+1]) if len(x) > i+2 else 0)
	    count_df[level] = count_df[level].map(lambda x : atlas_df[atlas_df["id"] ==x]["name"].values)
	    count_df[level] = count_df[level].map(lambda x : x[0] if len(x)>0 else "" )
	#count_df = count_df.drop(count_df[count_df["region"]==0].index)



	rowlist =[]

	levels =[]


	count_df = count_df.drop(columns = ["structure_id_path","structure_ids" ])
	for i in range(11):
	    level = "level_{}".format(10-i)
	    count_df = count_df.sort_values(level)
	    
	    
	    
	for i in range(nlevel):
	    level = "level_{}".format(i)
	    levels.append(level)
	df  = count_df.groupby(levels, as_index=False).sum()
	for i, row in df.iterrows():
	    
	    l = [level for level in levels if row[level] != ""]
	    if(len(l)==0):
	        continue
	    temp = df.groupby(l, as_index = False).sum()
	    for each in l:
	        temp = temp[temp[each] == row[each]]
	    
	    row["count"] = temp["count"].values[0]
	    rowlist.append(row)

	region_df = pd.DataFrame(rowlist)
	"""

	empty_df = pd.DataFrame(columns=["region","count"])
	empty_df["region"] =[""]
	empty_df["count"] =[""]

	df =  count_df.groupby("region name", as_index=False).sum()[["region name","count"]]
	df.columns = ["region","count"]



	pd_list =[df,empty_df]
	for i in range(11):
	    level = "level_{}".format(i)
	    df  = count_df.groupby(level, as_index=False).sum()[[level,"count"]]
	    df.columns = ["region","count"]
	    df = df.drop(df[df["region"]==""].index)
	    pd_list.append(df)
	    pd_list.append(empty_df)
	region_df = pd.concat(pd_list)
	"""
	return region_df, count_df


if __name__ == "__main__":
	args = arg_parser().parse_args()

	nlevel = int(args.level)
	csv_file = args.csv

	atlas_df = pd.read_csv(r"./CCF_DATA/1_adult_mouse_brain_graph_mapping.csv", index_col=None)
	count_df = pd.read_csv(csv_file, index_col=None)

	region_df,count_df = process_counts(atlas_df, count_df, nlevel)
	
	count_df.to_csv(os.path.join(args.out,"cell_region_count.csv"), index=False)
	region_df.to_csv(os.path.join(args.out,"region_counts.csv"), index=False)