import sys
import xgboost as xgb
import pandas as pd
import numpy as np

print("----reading data\n")
train = pd.read_csv("train.csv")
train_feature = train.columns[0:-1]
train_label = train.columns[-1]

print("----training a XGBoost\n")
dtrain = xgb.DMatrix(train[train_feature].values, label=train[train_label].values)
param = {'max_depth': 5,
         'eta': 1,
         'eval_metric': 'auc'}
bst = xgb.train(param, dtrain, 30)


print("----predict stock\n")
fi = open("test.csv", 'r')
fulldata = []
linenum=0
features_num = 500
fea_str =[]
for line in fi:
    if linenum%100==0: sys.stderr.write('%f\n' % linenum)
    linenum += 1
    try:
        features = line.strip("\n").split(",")
        data = []
        inx = 1
        for i in features.split(','):
            if inx > int(features_num):
                continue
            inx += 1
            data.append(float(i))

        fulldata.append(data)
        fea_str.append('%s' % '\t'.join(features))
    except Exception as e:
        sys.stderr.write('Exception: %s\n' % str(e))
        sys.stderr.write('wrong line: %s\n' % line)
        pass
xgb_input = np.array(fulldata)
label = np.array([-1])
test = xgb.DMatrix(xgb_input, label=label)
pred = bst.predict(test)

print("--- print result")
for fea_str, pred in zip(fea_str, pred):
    print(fea_str + '\t' + str(pred) + '\n')





