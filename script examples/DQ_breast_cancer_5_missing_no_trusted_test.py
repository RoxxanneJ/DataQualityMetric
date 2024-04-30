import pandas as pd
import data_quality_metric as dqm
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

cancer_m_5 = pd.read_csv("../dataset/cancer/missing/cancer_missing_5.csv")
cancer_m_5.dropna(inplace=True)

if __name__ == '__main__':
    freeze_support()
    dqm.dq_metric(cancer_m_5, crt_names, models, 'cancer', 'example_cancer_5_missing', nb_iter=30)
    # .npy files with the metric will be saved in the directory output/scores/, intermediate results are saved in the
    # directory output/base_scores/ (accuracies and f1 scores)
    # and in the directory output/variations/ (accuracies and f1 scores when 5% of the errors in crt_names are injected
    # in data)
