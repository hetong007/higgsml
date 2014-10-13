Higgs Machine Learning Competition model submission
=======

This model is created by [Tianqi Chen](https://github.com/tqchen) and [Tong He](https://github.com/hetong007).

This model achieved the best private score among all of our models without ensemble learning. The scores for public board and private board are 3.72181 and 3.72370 respectively. 

We integrate some physical features with the original features, then feed the new dataset to [xgboost](https://github.com/tqchen/xgboost) for training and prediction.

Notes
======
* The major physics features we add is the sum momentum, invariant mass, energy of arbitary subset of {lep, tau, jet_leading, jet_subleading}
* We adjust several parameters to avoid overfitting
   - eta is set to small value 0.01, which usually needs more rounds to converge, but make results more stable
   - min_child_weight is set to 100, which mean each leaf value requires at least 900 sum of weights, making leave weight estimation more stable
   - colsampleby_tree is set to 0.5, every iteration we randomly pick half of features to construct the tree, this speedup training, and sometimes helps avoid overfitting
   - gamma is set to 0.1, this is a prunning parameter, we didn't tune it carefully, but leaving it nonzero do helps, because the trees in later phase will tends to be simpler, making the boosting less easy to overfit

On a laptop with an 8-thread i7 CPU, this model will run in less than an hour with less than 2GB memory.
