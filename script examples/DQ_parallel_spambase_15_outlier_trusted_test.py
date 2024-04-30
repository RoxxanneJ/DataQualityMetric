import pandas as pd
import data_quality_metric as dqm
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

spambase_o_15 = pd.read_csv("../dataset/spambase/trusted_test/spambase_train_outlier_15.csv")
y_train = spambase_o_15['class'].copy()
spambase_o_15.drop(columns=['class'], inplace=True)
spambase_test = pd.read_csv("../dataset/spambase/trusted_test/spambase_test.csv")
y_test = spambase_test['class'].copy()
spambase_test.drop(columns=['class'], inplace=True)

if __name__ == '__main__':
    freeze_support()
    # parallel on the models training
    dqm.dq_metric_test_para(spambase_o_15, spambase_test, y_train, y_test, crt_names, models, 'spambase',
                            'example_spambase_15_outlier')
    # .npy files with the metric will be saved in the directory output/scores/, intermediate results are saved in the
    # directory output/base_scores/ (accuracies and f1 scores)
    # and in the directory output/variations/ (accuracies and f1 scores when 5% of the errors in crt_names are injected
    # in data)
