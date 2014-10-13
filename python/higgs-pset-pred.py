# this is the example script to use xgboost to train 
import os
import sys
import numpy as np
import inspect
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "../xgboost/wrapper")
sys.path.append(code_path)
import xgboost as xgb
import physics as phy

if len(sys.argv) < 4:
    print 'Usage: <test.csv> <model.dat> <submission.csv>'
    exit(-1)

dpath_test = sys.argv[1]
dpath_model = sys.argv[2]
dpath_result = sys.argv[3]

lc = 0.5
test_size = 550000
threshold_ratio = 0.15
outfile = sys.argv[1].rsplit('.',1)[0]+".csv"
print outfile
# path to where the data lies

idx, dtest, punit, pset = phy.load_test(dpath_test)
# list of features that we want
features = set(['E_inv', 'E_tri', 'm_tri', 'm_inv', 'pts', 'p_x', 'p_y', 'p_z'])

# use all features without met for now
dpset = phy.mkf_pset([p for p in pset], features)

# concatenate all features together
dtest = np.concatenate([dtest, dpset], axis=-1)
print 'finish making features, shape=%s' % str(dtest.shape)

# print weight statistics 
xgmat = xgb.DMatrix(dtest, missing = -999.0)
bst = xgb.Booster()
bst.load_model(dpath_model)
ypred = bst.predict( xgmat )

res  = [ ( int(idx[i]), ypred[i] ) for i in range(len(ypred)) ] 
rorder = {}
for k, v in sorted( res, key = lambda x:-x[1] ):
    rorder[ k ] = len(rorder) + 1
# write out predictions
ntop = int( threshold_ratio * len(rorder ) )
fo = open(dpath_result, 'w')
nhit = 0
ntot = 0
fo.write('EventId,RankOrder,Class\n')
for k, v in res:        
    if rorder[k] <= ntop:
        lb = 's'
        nhit += 1
    else:
        lb = 'b'        
    # change output rank order to follow Kaggle convention
    fo.write('%s,%d,%s\n' % ( k,  len(rorder)+1-rorder[k], lb ) )
    ntot += 1
fo.close()
print ('finished writing into prediction file')
