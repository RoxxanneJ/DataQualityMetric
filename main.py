import pandas as pd
import data_quality_metric as dqm
from multiprocessing import freeze_support

# Select the models to use
models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']

# Select the types of errors to consider
crt_names = ['missing', 'fuzzing', 'outlier']

# Load data

if __name__ == '__main__':
    freeze_support()
    # Compute DQ with the appropriate version of the function (refer to the example scripts)
    # Read results (refer to the example scripts)
