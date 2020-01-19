from config import Config
from dataset import train_set,valid_set,test_set,test_valid_set
from dynamic_dataset import d_train_set,d_valid_set
from model import static_network# ,dynamic_network
from sklearn import metrics
import numpy as np
from list2df import list2df,test_format

def train(config):
    if config.model_type=='static':
        train_data = train_set(config)
        valid_data = valid_set(config)
    else:
        train_data=d_train_set(config)
        valid_data=d_valid_set(config)
    print('data loaded!')

    if config.model_type=='static':
        model = static_network(config)
    # else:
    #     model=dynamic_network(config)
    model.train(train_data, valid_data)

def test(config):
    test_data = test_set(config)
    test_data.re_init()
    print('data loaded!')

    model = static_network(config)
    output_list=model.test(test_data, choose_id=config.valid_list)
    df=list2df(output_list[0],output_list[1],output_list[2],output_list[3],output_list[4])
    df.to_csv("./submit/subchallenge_1_data_reindex.csv")
    test_format()

def valid(config):
    valid_data=test_valid_set(config)
    valid_num=valid_data.data_num
    ori_data=valid_data.valid_ori
    print(valid_num)
    print('data loaded!')

    model = static_network(config)
    valid_data.re_init()
    output_list=model.test(valid_data, choose_id=config.valid_list)

    print(output_list[0])
    for idx,i in enumerate(config.valid_list):
        print(ori_data[:,i])
        RMSE = np.sqrt(metrics.mean_squared_error(ori_data[:,i], output_list[idx]))
        print(RMSE)

if __name__ == '__main__':
    configs = Config()
    if configs.mode=='test':
        test(configs)
    elif configs.mode=='valid':
        valid(configs)
    else:
        train(configs)












