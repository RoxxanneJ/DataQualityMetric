import pandas as pd
import data_quality_metric as dqm
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

# the dataset iris with 25% of a combination of missing values and outliers injected
train_deteriorated = pd.read_csv("dataset/iris/trusted_test/iris_train_missing_outlier_25.csv")
y_train_deteriorated = train_deteriorated["class"].copy()
train_deteriorated.drop(columns=["class"], inplace=True)

# the dataset iris when the errors injected have been repaired with 1. replacing missing values with attributes' means
# and 2. detecting 10% of outliers with an isolation forest and deleting them
train_repaired = pd.read_csv("dataset/iris/trusted_test/iris_train_m")
y_train_repaired = train_repaired["class"].copy()
train_repaired.drop(columns=["class"], inplace=True)

# trusted test set without errors
test = pd.read_csv("dataset/iris/trusted_test/clean/iris_train_missing_outlier_25_cleaned.csv")
y_test = test["class"].copy()
test.drop(columns=["class"], inplace=True)

if __name__ == '__main__':
    freeze_support()
    dqm.dq_metric_test_para(train_deteriorated, test, y_train_deteriorated, y_test, crt_names, models,
                            'iris', 'iris_deteriorated_example')
    dqm.dq_metric_test_para(train_repaired, test, y_train_repaired, y_test, crt_names, models,
                            'iris', 'iris_repaired_example')
    _, qa_det, _, _ = pd.read_csv("output/scores/iris_deteriorated_example_(x,qa,qf,time).npy")
    perf_deteriorated = 1 - qa_det[1]  # we use 1-qa1
    _, qa_rep, _, _ = pd.read_csv("output/scores/iris_repaired_example_(x,qa,qf,time).npy")
    perf_repaired = 1 - qa_rep[1]

    rep_deg = rp.repairability_degree(perf_deteriorated, perf_repaired)
    print(rep_deg)
