import pandas as pd
import data_quality_metric as dqm
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

statlog_o_10 = pd.read_csv("../dataset/statlog/trusted_test/statlog_train_outlier_10.csv")
y_train = statlog_o_10['class'].copy()
statlog_o_10.drop(columns=['class'], inplace=True)
statlog_test = pd.read_csv("../dataset/statlog/trusted_test/statlog_test.csv")
y_test = statlog_test['class'].copy()
statlog_test.drop(columns=['class'], inplace=True)

if __name__ == '__main__':
    freeze_support()
    dqm.dq_metric_test(statlog_o_10, statlog_test, y_train, y_test, crt_names, models, 'statlog',
                       'example_statlog_10_outlier')
    # .npy files with the metric will be saved in the directory output/scores/, intermediate results are saved in the
    # directory output/base_scores/ (accuracies and f1 scores)
    # and in the directory output/variations/ (accuracies and f1 scores when 5% of the errors in crt_names are injected
    # in data)
