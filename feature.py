import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import operator
import os

# path = './synapse/complete_cell_line/'
path = './train/'
test_path = './synapse/subchallenge/subchallenge1/'
template_path = './synapse/template/'
out_path = './output/'
trg_id = ['p.ERK', 'p.Akt.Ser473.', 'p.S6', 'p.HER2', 'p.PLCg2']


# dev_id = ['p.S6K', 'p.SMAD23', 'p.STAT1']


def feature_tag(tag_list, label_tag):
    feature_list = []
    for i in tag_list:
        if i not in label_tag:
            feature_list.append(i)
    print("features to be used:", feature_list)
    return feature_list


def read_complete_cellline(path):
    csv_list = []
    for csv in os.listdir(path):
        csv_list.append(csv)
    for idx, csv in enumerate(csv_list):
        if idx == 0:
            print('train_file:', csv, end='\t')
            complete_data = pd.read_csv(path + csv)
        else:
            reader = pd.read_csv(path + csv, encoding='utf-8')
            complete_data = pd.concat([complete_data, reader])
            print(csv, end='\t')
    print("finish loading complete cell line data:\n", complete_data.head())
    print("start random shuffle...\n")
    # complete_data.to_csv("temp.csv", index=False)
    return {"complete_data": complete_data, "length": len(complete_data)}


def read_train_file(path):
    csv_list = []
    file_num = len(os.listdir(path))
    train_data_num = int((file_num / 10) * 7)
    for csv in os.listdir(path):
        csv_list.append(csv)
    for id, csv in enumerate(csv_list):
        if id < train_data_num:
            if id == 0:
                print('train_file:', csv, end='\t')
                train_data = pd.read_csv(path + csv)
            else:
                reader = pd.read_csv(path + csv, encoding='utf-8')
                train_data = pd.concat([train_data, reader])
                print(csv, end='\t')
        else:
            if id == train_data_num:
                print('\ndev_file:', csv, end='\t')
                dev_data = pd.read_csv(path + csv)
            else:
                reader = pd.read_csv(path + csv)
                dev_data = pd.concat([dev_data, reader])
                print(csv, end='\t')
        # print(id,end='\t')
        # if id==0:break
    return {"train_data": train_data, "dev_data": dev_data}


def read_test_file(test_path):
    test_csv_list = []
    for csv in os.listdir(test_path):
        test_csv_list.append(csv)
    for id, csv in enumerate(test_csv_list):
        if id == 0:
            test_data = pd.read_csv(test_path + csv)
        else:
            reader = pd.read_csv(test_path + csv)
            test_data = pd.concat([test_data, reader])
    return test_data


def get_columns(columns):
    col_list = []
    for col in columns:
        if col != 'treatment' and col != 'cell_line' and col != 'cellID' and col != 'fileID' and col != 'index':
            col_list.append(col)
    print("columns to be used:", col_list)
    return col_list


def split_by_condition(condition_list, data):
    dp_split = []
    for idx, condition in enumerate(condition_list):
        # for test,only consider one condition
        print(condition)
        dp_split.append(data[data['treatment'].isin([condition])])
    return dp_split


def train_model(mode='lgb'):
    trg_id = ['p.ERK', 'p.Akt.Ser473.', 'p.S6', 'p.HER2', 'p.PLCg2']

    # Load data from .csv file
    # load_data = readTrainFile(path)
    load_data = read_complete_cellline(path)
    # load_data = pd.read_csv("temp.csv")
    print('data loading completed!')
    # complete_data = load_data.get("complete_data")
    complete_data = load_data
    complete_data = shuffle(complete_data).reset_index()
    print("finish random shuffle complete cell line data:\n", complete_data.head())
    print("the length of complete cell line data:\n", len(complete_data))
    # complete_data_length = load_data.get("length")
    complete_data_length = len(complete_data)
    len_set = complete_data_length // 5  # 分成5个子集
    print("length of complete data: " + str(complete_data_length) + " lenth of every set: " + str(len_set))

    # get list of condition name
    condition_list = complete_data['treatment'].unique()

    # get list of column name, drop unwanted columns
    columns = complete_data.columns
    col_list = get_columns(columns)

    # for model_id in range(0, 5):
    for model_id in [0]:
        dev_index = range(model_id * len_set, (model_id + 1) * len_set + 1)
        dev_data = complete_data[model_id * len_set: (model_id + 1) * len_set + 1]
        train_data = complete_data.drop(dev_index)
        print("length of train data: " + str(len(train_data)) + " length of dev data: " + str(len(dev_data)))
        # split data by condition (treatment)
        dp_train_split = split_by_condition(condition_list, train_data)
        dp_dev_split = split_by_condition(condition_list, dev_data)
        print("finish split train_data and dev_data by condition")

        # train model_backup for different conditions
        recodes = ''
        for idx, condition in enumerate(condition_list):

            print("get data of current condition: ", condition)
            dp_train_idx = dp_train_split[idx]
            dp_dev_idx = dp_dev_split[idx]

            print("filter columns: unwanted columns, such as 'cellID'")
            dp_train = dp_train_idx[col_list]
            dp_dev = dp_dev_idx[col_list]

            for trg in trg_id:
                print("filter columns: markers to predict:", trg_id)
                x_train = dp_train[feature_tag(col_list, trg_id)].fillna(0).values
                x_dev = dp_dev[feature_tag(col_list, trg_id)].fillna(0).values
                y_train = dp_train[trg].fillna(0).values
                y_dev = dp_dev[trg].fillna(0).values
                print("start training:" + condition + trg + '.m')
                model_path = './model/' + mode + '/' + str(model_id) + '/' + condition + trg + '.m'

                if mode == 'rf':
                    rf = RandomForestRegressor()
                    if os.path.exists(model_path):
                        rf = joblib.load(model_path)
                    else:
                        rf.fit(x_train, y_train)
                        joblib.dump(rf, model_path)
                    y_pred = rf.predict(x_dev)
                elif mode == 'lgb':
                    # lgb_train = lgb.Dataset(x_train, y_train)
                    # lgb_eval = lgb.Dataset(x_dev, y_dev, reference=lgb_train)
                    # gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
                    gbm = lgb.LGBMRegressor(objective='regression', num_leaves=21, learning_rate=0.01,
                                            n_estimators=1000000)
                    if os.path.exists(model_path):
                        gbm = joblib.load(model_path)
                    else:

                        gbm.fit(x_train, y_train, eval_set=[(x_dev, y_dev)], eval_metric='rmse',
                                early_stopping_rounds=10)
                        joblib.dump(gbm, model_path)

                    y_pred = gbm.predict(x_dev, num_iteration=gbm.best_iteration_)

                RMSE = np.sqrt(metrics.mean_squared_error(y_dev, y_pred))
                recodes += str(idx) + condition + '\t' + trg + '\t' + str(RMSE) + '\t'
                print(condition + '\t' + trg + '\t', RMSE, '\t')

    print(recodes)


def get_predict(mode, model_id, trg_id, test_path, out_path):
    test_csv_list = []
    for csv in os.listdir(test_path):
        test_csv_list.append(csv)
    print('test_csv_list: ', test_csv_list)

    for id, csv in enumerate(test_csv_list):
        print('start processing:\t', 'id:', id, '\t', csv)
        test_data = pd.read_csv(test_path + csv)
        rs_col = ['cell_line', 'treatment', 'time', 'cellID', 'fileID']
        rs = pd.DataFrame(columns=rs_col)

        # get list of column name, drop unwanted columns
        columns = test_data.columns
        col_list = get_columns(columns)

        condition_list = test_data['treatment'].unique()
        print("condition_list:", condition_list)
        dp_test_split = split_by_condition(condition_list, test_data)

        for idx, condition in enumerate(condition_list):
            print(idx)
            dp_test_idx = dp_test_split[idx]
            dp_test = dp_test_idx[col_list]
            x_test = dp_test[feature_tag(col_list, trg_id)].fillna(0)
            print('data loading completed!')
            dp_result = dp_test_idx[rs_col]
            for trg in trg_id:
                if mode == 'rf':
                    # rf = RandomForestRegressor()
                    model_path = './model/' + mode + '/' + str(model_id) + '/' + condition + trg + '.m'
                    # x_train, x_test, y_train, y_test = train_test_split(X, Y)
                    if os.path.exists(model_path):
                        rf = joblib.load(model_path)
                    else:
                        print("Oops! No suitable model:" + model_path)
                        break
                    y_test_pred = rf.predict(x_test)
                elif mode == 'lgb':
                    # lgb_train = lgb.Dataset(x_train, y_train)
                    # lgb_eval = lgb.Dataset(x_dev, y_dev, reference=lgb_train)
                    # gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
                    # gbm = lgb.LGBMRegressor(objective='regression', num_leaves=21, learning_rate=0.01, n_estimators=1000000)
                    model_path = './model/' + mode + '/' + str(model_id) + '/' + condition + trg + '.m'
                    if os.path.exists(model_path):
                        gbm = joblib.load(model_path)
                    else:
                        print("Oops! No suitable model:" + model_path)
                        break
                    y_test_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration_)

                dp_result[trg] = y_test_pred
                print(condition + '\t' + trg + '\t', y_test_pred)
            rs = rs.append(dp_result)

        rs.describe()
        rs_df = pd.DataFrame(rs)
        rs_df.to_csv(out_path + str(model_id) + '/' + csv)


def test_format():
    #col = ["glob_cellID", "cell_line", "treatment", "time", "cellID", "fileID", "p.ERK", "p.Akt.Ser473.", "p.S6","p.HER2", "p.PLCg2"]
    tp = pd.read_csv("./synapse/template/subchallenge_1_template_data.csv")
    rs = pd.read_csv("./output/subchallenge_1_data.csv")
    # rs = rs[col]
    if not operator.eq(len(tp), len(rs)):
        print("ERROR! Different length.")
    if not all(operator.eq(tp.columns, rs.columns)):
        print("ERROR! Different columns.")
    # print("template head:", tp.head())
    # print("rs head':", rs.head())
    # print("template tail:", tp.tail())
    # print("rs tail:", rs.tail())
    if not all(operator.eq(tp.head(), rs.head())):
        print("ERROR! unmatched head")
    if not all(operator.eq(tp.tail(), rs.tail())):
        print("ERROR! unmatched tail")
    if not all(tp['glob_cellID'] == rs['glob_cellID']):
        print(rs[(tp['glob_cellID'] == rs['glob_cellID']) == False].head())
        print(tp[(tp['glob_cellID'] == rs['glob_cellID']) == False].head())
        print("ERROR! unmatched glob_cellID")
    if not all(tp['cell_line'] == rs['cell_line']):
        print("ERROR! unmatched cell_line")

    if not all(tp['treatment'] == rs['treatment']):
        print("ERROR! unmatched treatment")
    if not all(tp['time'] == rs['time']):
        print("ERROR! unmatched time")
    if not all(tp['cellID'] == rs['cellID']):
        print("ERROR! unmatched cellID")
    if not all(tp['fileID'] == rs['fileID']):
        print("ERROR! unmatched fileID")
    print("Format check: OK!")



def sub1_predict_file(model_id):
    path = './output/'+str(model_id)+'/'
    col = ["cell_line", "treatment", "time", "cellID", "fileID", "p.ERK", "p.Akt.Ser473.", "p.S6", "p.HER2", "p.PLCg2"]
    csv_list = []
    for csv in os.listdir(path):
        csv_list.append(csv)
    for id, csv in enumerate(csv_list):
        if id == 0:
            data = pd.read_csv(path + csv)[col]
        else:
            reader = pd.read_csv(path + csv)[col]
            data = pd.concat([data, reader])
    print(model_id,len(data))
    data.to_csv('./output/'+str(model_id)+'.csv', index=False)

def average():
    col2 = ["cell_line", "treatment", "time", "cellID", "fileID"]
    col = ["p.ERK", "p.Akt.Ser473.", "p.S6", "p.HER2", "p.PLCg2"]
    flag = True
    for i in range(5):
        path = './output/' + str(i) + '.csv'
        if flag:
            m = pd.read_csv(path)[col].values
            flag = False
        else:
            m = m + pd.read_csv(path)[col].values
    m = m / 5
    m = pd.DataFrame(m)
    m.columns = col
    m_info = pd.read_csv('./output/0.csv').loc[:, col2]
    fn = pd.concat([m_info, m], axis=1)
    fn.to_csv('./output/'+'average_predict.csv', index=False)

def template_generate():
    tp = pd.read_csv("./synapse/template/subchallenge_1_template_data.csv")
    df = pd.read_csv("./output/average_predict.csv")
    col = ["cell_line", "treatment", "time", "cellID", "fileID"]
    col1 = ["cell_line", "treatment", "time", "cellID", "fileID", "p.ERK", "p.Akt.Ser473.", "p.S6", "p.HER2", "p.PLCg2"]
    col2 = ["glob_cellID", "cell_line", "treatment", "time", "cellID", "fileID", "p.ERK", "p.Akt.Ser473.", "p.S6",
            "p.HER2", "p.PLCg2"]
    tp = tp[col]
    rs = pd.merge(tp, df, how='left', on=col)
    print(len(tp), len(df), len(rs))
    print(df.head())
    rs = rs[col1]
    rs["glob_cellID"] = np.arange(1, len(rs) + 1)
    #
    rs = rs[col2]
    rs.reset_index()
    rs.to_csv("./output/subchallenge_1_data.csv", index=False)





# train_model()
# read_complete_cellline(path)
# get_predict('lgb', 0, trg_id, test_path, out_path)
# test_format()
# sub1_predict_file(0)
# average()
# template_generate()
test_format()
