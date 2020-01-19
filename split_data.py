import os
import random
import pandas as pd
label_dict={'EGF':0, 'full':1, 'iEGFR':2,'iMEK':3, 'iPI3K':4, 'iPKC':5}
data_file='./data/complete/train/'
refector_data_file= './data/complete/split/'
for key in label_dict.keys():
    if not os.path.exists(refector_data_file + key):
        os.mkdir(refector_data_file + key)
        print(refector_data_file + key)

def splitByCondition(condition_list,col_string, data):
    dp_split = []
    for idx, condition in enumerate(condition_list):
        # for test,only consider one condition
        dp_split.append(data[data[col_string].isin([condition])])
    return dp_split

def build_time_line(data,name):
    data_num=data.shape[0]
    condition_list = data['treatment'].unique()
    data_split=splitByCondition(condition_list,'treatment',data)
    output_data=[]
    for idx,id in enumerate(condition_list):
        condition_data=data_split[idx]
        # print(condition_data)
        time_line_list=condition_data['time'].unique()
        time_split=splitByCondition(time_line_list,'time',condition_data)
        for time_idx,time in enumerate(time_line_list):
            # print(time_split[time_idx])
            time_data=time_split[time_idx]
            time_data.to_csv(refector_data_file+id+'/'+name+'_'+str(time_idx),index=False)
            print(id,time)
for csv in os.listdir(data_file):
    name=csv.split('.')[0]
    csv=data_file+csv
    data=pd.read_csv(csv)
    build_time_line(data,name)
