# this is the example script to use xgboost to train 
import os
import sys
import numpy as np
# add path of xgboost python module
sys.path.append("../xgboost/wrapper/")
import xgboost as xgb
import physics as phy

if len(sys.argv) < 3:
    print 'Usage: <train.csv> <model.dat>'
    exit(-1)

dpath_train = sys.argv[1]
dpath_model = sys.argv[2]

eta = 0.01
nround = 3000
lc = 0.5
test_size = 550000

label, dtrain, weight, punit, pset = phy.load_train(dpath_train)
# list of features that we want
features = set(['E_inv', 'E_tri', 'm_tri', 'm_inv', 'pts', 'p_x', 'p_y', 'p_z'])
# use all features without met for now
dextra = phy.mkf_pset([p for p in pset], features)

# concatenate all features together
dtrain = np.concatenate([dtrain, dextra], axis=-1)

print 'finish making features, shape=%s' % str(dtrain.shape)

# rescale weight to make it same as test set
weight = weight * float(test_size) / len(label)

sum_wpos = sum(weight[i] for i in range(len(label)) if label[i] == 1.0)
sum_wneg = sum(weight[i] for i in range(len(label)) if label[i] == 0.0)
# print weight statistics 
print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))

# construct xgboost.DMatrix from numpy array, treat -999.0 as missing value
xgmat = xgb.DMatrix(dtrain, label=label, missing = -999.0, weight=weight)

# setup parameters for xgboost
param = {}
# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
param['objective'] = 'binary:logitraw'
# scale weight of positive examples
param['scale_pos_weight'] = sum_wneg/sum_wpos
param['bst:eta'] = eta
param['eval_metric'] = 'auc'
param['silent'] = 1
param['bst:min_child_weight'] = 100
param['bst:max_depth'] = 9
param['bst:col_samplebytree'] = lc
param['bst:gamma']=0.1

# boost 120 tres
num_round = nround
# you can directly throw param in, though we want to watch multiple metrics here 

watchlist = [ (xgmat,'train') ]
print ('loading data end, start to boost trees')
bst = xgb.train( param, xgmat, num_round, watchlist );
# save out model
bst.save_model(dpath_model)

