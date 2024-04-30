from sklearn.datasets import load_iris
import pandas as pd
import data_quality_metric as dqm
from multiprocessing import freeze_support
import numpy as np

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

abalone_m_5 = pd.read_csv("dataset/abalone/missing/abalone_missing_5.csv")

if __name__ == '__main__':
    freeze_support()
    v = np.load("output/scores/abalone_missing_5_noTest_(x,qa,qf,time).npy", allow_pickle=True)
