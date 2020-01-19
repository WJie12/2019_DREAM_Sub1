import os
import pandas as pd
import numpy as np
import random

class treatment_encoder():
    def __init__(self):
        self.label_dict={'EGF':0, 'full':1, 'iEGFR':2,
                         'iMEK':3, 'iPI3K':4, 'iPKC':5}
        self.num_dict={0:'EGF', 1:'full', 2:'iEGFR',
                       3:'iMEK', 4:'iPI3K', 5:'iPKC'}
    def __add__(self, item):
        self.num_dict[len(self.label_dict)]=item
        self.label_dict[item]=len(self.label_dict)
    def encode(self,item):
        return self.label_dict[item]
    def decode(self,num):
        return self.num_dict[num]

def rebuild_data(data,treatment_label=True,time=True):
    columns = data.columns
    # print(columns)
    col_list = []
    for col in columns:
        if  col != 'cell_line' and col != 'cellID' and col != 'fileID':
            col_list.append(col)
    if treatment_label==False:
        col_list.remove('treatment')
    if time==False:
        col_list.remove('time')
    data = data[col_list]
    if treatment_label:
        treatment_enc = treatment_encoder()
        for key in treatment_enc.label_dict:
            data['treatment'] = data['treatment'].replace(key, treatment_enc.label_dict[key])
    data = data.fillna(data.sum() / data.shape[0])
    return data.values

# the first step is get all cell line names from train
# dir='./data/test/train/'
# csv='184A1.csv'
def get_all_csv(ori_dir):
    csv_list=[]
    for csv in os.listdir(ori_dir):
        csv_list.append(csv)
    return csv_list
# save dataset
# split_dir='./data/test/split/'
# max_length=6
# condition='iMEK'
def file_list_generate(split_dir,csv,condition,max_length):
    file_list=[]
    name = csv.split('.')[0]
    for i in range(max_length):
        file=name+'_'+str(i)
        csv_file=split_dir+condition+'/'+file
        if os.path.exists(csv_file):
            file_list.append(csv_file)
        else:
            file_list.append([file_list[i-1]])
    return file_list

def time_line_generate(file_list,max_length):
    data_list=[]
    output_list=[]
    for files in file_list:
        data=pd.read_csv(files)
        # data=data.fillna(data.sum() / data.shape[0])
        data=rebuild_data(data)
        data_list.append(data)
    train_num=data_list[0].shape[0]
    for _ in range(train_num):
        batch_list=[]
        for i in range(max_length):
            random_idx = random.randint(0, data_list[i].shape[0] - 1)
            batch_list.append(data_list[i][random_idx])
        output_list.append(np.array(batch_list))
    return np.array(output_list)

def data_generate(split_dir,csv_list,condition,max_length):
    data_list=[]
    for csv in csv_list:
        file_list=file_list_generate(split_dir,csv,condition,max_length)
        csv_data=time_line_generate(file_list,max_length)
        data_list.append(csv_data)
    return np.array(data_list)


class d_train_set():
    def __init__(self,config):
        self.config=config
        self.idx=0
        self.batch_size=config.batch_size
        self.csv_list=get_all_csv(config.ori_dir)
        self.csv_list=self.csv_list[:config.train_num]
        data_path=config.split_dir
        self.train_data=data_generate(data_path,self.csv_list,config.condition,config.max_length)
        self.train_data=self.train_data.reshape((-1,config.max_length,config.gene_num+2))
        # self.train_data=self.train_data.fillna(self.train_data.sum()/self.train_data.shape[0])
        self.train_num=self.train_data.shape[0]

    def re_init(self):
        self.idx = 0
        np.random.shuffle(self.train_data)

    def __iter__(self):
        return self

    def __len__(self):
        return self.train_data.shape[0]

    def __next__(self):
        if self.idx + self.batch_size > self.train_num:
            raise StopIteration
        else:
            self.idx += self.batch_size
            return self.train_data[self.idx - self.batch_size:self.idx]


# from config import Config
# data=train_set(Config())
# print('data load!')
class d_valid_set():
    def __init__(self,config):
        self.config = config
        self.idx = 0
        self.batch_size = config.batch_size
        self.csv_list = get_all_csv(config.ori_dir)
        self.csv_list=self.csv_list[config.train_num:]
        data_path = config.split_dir
        self.valid_data = data_generate(data_path, self.csv_list, config.condition, config.max_length)
        self.valid_data=self.valid_data.reshape((-1,config.max_length,config.gene_num+2))

        self.valid_num = self.valid_data.shape[0]

    def re_init(self):
        self.idx=0
        np.random.shuffle(self.valid_data)

    def __iter__(self):
        return self
    def __len__(self):
        return self.valid_data.shape[0]

    def __next__(self):
        if self.idx+self.batch_size>self.valid_num:
            raise StopIteration
        else:
            self.idx+=self.batch_size
            return self.valid_data[self.idx-self.batch_size:self.idx]
