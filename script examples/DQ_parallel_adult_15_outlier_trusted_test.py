import pandas as pd
import data_quality_metric as dqm
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

adult_o_15 = pd.read_csv("dataset/adult_train_outlier_15.csv")
y_train = adult_o_15['class'].copy()
adult_o_15.drop(columns=['class'], inplace=True)
adult_test = pd.read_csv("dataset/adult_test.csv")
y_test = adult_test['class'].copy()
adult_test.drop(columns=['class'], inplace=True)

if __name__ == '__main__':
    freeze_support()
    # parallel on the models training
    dqm.dq_metric_test_para(adult_o_15, adult_test, y_train, y_test, crt_names, models, 'adult',
                            'example_adult_15_outlier')
    # .npy files with the metric will be saved in the directory output/scores/, intermediate results are saved in the
    # directory output/base_scores/ (accuracies and f1 scores)
    # and in the directory output/variations/ (accuracies and f1 scores when 5% of the errors in crt_names are injected
    # in data)
