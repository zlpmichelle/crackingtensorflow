#!/usr/bin/env python
import xgboost as xgb
import pandas as pd
import numpy as np
import copy
import sys


def load_data():

    train_data = pd.read_csv(os.path.join(data_folder, 'train.csv'), delimiter=';', skip_blank_lines=True)
    test_data = pd.read_csv(os.path.join(data_folder, 'test.csv'), delimiter=';', skip_blank_lines=True,
                            na_values='None')

    ntrain = train_data.shape[0]
    ntest = test_data.shape[0]

    print('ntrain={}'.format(ntrain))
    print('ntest={}'.format(ntest))

    y_train = train_data['cardio'].values

    # --------------------------------------------------------------

    x_train = train_data.drop(["id", "cardio"], axis=1)
    x_test = test_data.drop(["id"], axis=1)

    x_test.replace('None', np.nan)

    return (x_train,y_train,x_test)

def single_run(num_round, param, data, model):
    """
    Run GBDT giving parameters
    Input:
        param: input_param
    """

    train = xgb.DMatrix(data, silent=1)

    param['silent'] = 1
#    param['objective'] = 'binary:logistic'
    param['scale_pos_weight'] = 1.0
#    param['booster'] = 'gblinear'
    bst = xgb.train(param, train, num_round)
    bst.save_model(model)
    bst.dump_model('model.txt')
    importance = pd.DataFrame(bst.get_fscore().items(), columns=['feature','importance'])
    importance.sort('importance', ascending=False, inplace=True)
    print importance

def run_gbdt(data, model):
    """
    Run gbdt multiple times
    """
    param = {}
    param['max_depth'] = 6
    param['colsample_bytree'] = 0.8
    # param['max_delta_step'] = 5.0
    num_round = 50
    single_run(num_round, param, data, model)

if __name__ == '__main__':
    run_gbdt( sys.argv[1], sys.argv[2])
