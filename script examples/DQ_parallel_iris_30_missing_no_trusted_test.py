import pandas as pd
import data_quality_metric as dqm
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

iris_m_30 = pd.read_csv("../dataset/iris/missing/iris_missing_30.csv")
iris_m_30.dropna(inplace=True)

if __name__ == '__main__':
    freeze_support()
    # parallel on the 30 resamplings
    dqm.dq_metric_para(30, iris_m_30, crt_names, models, 'iris', 'example_iris_30_missing')
    # .npy files with the metric will be saved in the directory output/scores/, intermediate results are saved in the
    # directory output/base_scores/ (accuracies and f1 scores)
    # and in the directory output/variations/ (accuracies and f1 scores when 5% of the errors in crt_names are injected
    # in data)
