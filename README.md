Higgs Machine Learning Competition model submission
=======

This model is created by [Tianqi Chen](https://github.com/tqchen) and [Tong He](https://github.com/hetong007).

This model achieved the best private score among all of our models without ensemble learning. The scores for public board and private board are 3.72181 and 3.72370 respectively. 

We integrate some physical features with the original features, then feed the new dataset to [xgboost](https://github.com/tqchen/xgboost) for training and prediction.

On a laptop with an 8-thread i7 CPU, this model will run in less than an hour with less than 2GB memory.
