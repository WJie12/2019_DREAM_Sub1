import os
import pandas as pd
import numpy as np
import copy
from random import shuffle

def padding(data,batch_size):
    length=len(data)
    last_data=data[length-1]
    padding_num=length%batch_size
    for i in range(batch_size-padding_num):
        data=np.row_stack((data, last_data))
    return data,padding_num


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

def read_datas(dir,split=None,revers=True):
    files_list=[]
    if split!=None:
        if revers==False:
            for idx,file in enumerate(os.listdir(dir)):
                files_list.append(dir+file)
                if idx==split-1:break
        else:
            for idx,file in enumerate(os.listdir(dir)):
                if idx>=split:
                    files_list.append(dir+file)
    else:
        for idx, file in enumerate(os.listdir(dir)):
            files_list.append(dir + file)
    return files_list

def read_pd(files,file_counter=False,time=False):
    file_length=[]
    # print(files)
    for id, csv in enumerate(files):
        if id==0:
            data=rebuild_data(pd.read_csv(csv),time=time)
            data=data.fillna(data.sum()/data.shape[0])
            data=data.values
            if file_counter:
                file_length.append(data.shape[0])
        else:
            reader = pd.read_csv(csv,encoding='utf-8')
            reader=rebuild_data(reader,time=time)
            reader=reader.fillna(reader.sum()/reader.shape[0])
            reader=reader.values
            data=np.vstack((data,reader))
            if file_counter:
                file_length.append(reader.shape[0])
        print(csv,end=' ')
    if file_counter:
        return data,file_length
    else:return data

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
    return data

class train_set():
    def __init__(self,config):
        self.config=config
        self.idx=0
        self.batch_size=config.batch_size
        self.dataset=read_datas(config.ori_train,split=config.train_num,revers=False)
        train_files=self.dataset
        self.train_data=read_pd(train_files,time=config.use_position_embedding)
        self.train_num=self.train_data.shape[0]

    def re_init(self):
        self.idx=0
        np.random.shuffle(self.train_data)
    def __iter__(self):
        return self
    def __len__(self):
        return self.train_data.shape[0]
    def __next__(self):
        if self.idx+self.batch_size>self.train_num:
            raise StopIteration
        else:
            self.idx+=self.batch_size
            return self.train_data[self.idx-self.batch_size:self.idx]

class valid_set():
    def __init__(self,config):
        self.config = config
        self.idx = 0
        self.batch_size = config.batch_size
        self.dataset = read_datas(config.ori_train,split=config.train_num,revers=True)
        valid_files=self.dataset
        self.valid_data=read_pd(valid_files,time=config.use_position_embedding)
        self.valid_num=self.valid_data.shape[0]
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

class test_valid_set():
    def __init__(self,config):
        self.config = config
        self.idx = 0
        self.batch_size = config.batch_size
        self.dataset = read_datas(config.ori_train,split=config.train_num,revers=True)
        valid_files=self.dataset
        self.valid_data=read_pd(valid_files)
        self.data_num=self.valid_data.shape[0]
        self.valid_ori=copy.deepcopy(self.valid_data)


    def re_init(self):
        self.idx=0
        self.valid_data[:, self.config.valid_list] = 0
        self.valid_data, self.padding_num = padding(self.valid_data, batch_size=self.batch_size)

    def __iter__(self):
        return self
    def __len__(self):
        return self.valid_data.shape[0]

    def __next__(self):
        if self.idx + self.batch_size > self.data_num+self.batch_size:
            raise StopIteration
        else:
            self.idx += self.batch_size
            return self.valid_data[self.idx - self.batch_size:self.idx]



class test_set():
    def __init__(self,config):
        self.config = config
        self.idx = 0
        self.batch_size = config.batch_size
        self.dataset = read_datas(config.ori_test,revers=True)
        test_files = self.dataset
        self.test_data = read_pd(test_files)
        self.data_num = self.test_data.shape[0]
    def re_init(self):
        self.idx=0
        self.test_data[:, self.config.valid_list] = 0
        self.test_data, self.padding_num = padding(self.test_data, batch_size=self.batch_size)

    def __iter__(self):
        return self

    def __len__(self):
        return self.test_data.shape[0]

    def __next__(self):
        if self.idx + self.batch_size > self.data_num + self.batch_size:
            raise StopIteration
        else:
            self.idx += self.batch_size
            return self.test_data[self.idx - self.batch_size:self.idx]
