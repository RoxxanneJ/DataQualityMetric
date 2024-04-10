# Data Quality Evaluation

## About

Abstract + link to paper to add


## Table of contents

> [Data Quality Evaluation](#Data-Quality-Evaluation)
> * [About](#about)
> * [Table of contents](#table-of-contents)
> * [Architecture of the tool](#architecture-of-the-tool)
> * [Usage](#usage)
>  * [Requirements](#requirements)
>  * [Examples](#examples)
>    * [With dedicated test data](#with-dedicated-test-data)
>      * [Not parallel (test data)](#not-parallel-test-data)
>      * [Parallel (test data)](#parallel-test-data)
>    * [Without dedicated test data](#without-dedicated-test-data)
>      * [Not parallel](#not-parallel)
>      * [Parallel](#parallel)
## Architecture of the tool

![image info](./schema-data-quality-tool.png)

## Computing data quality
### Requirements
The classification target needs to be called 'class' in the dataframe

###Examples:
###With dedicated test data:
####Not parallel (test data):
```python
from sklearn.datasets import load_iris
import pandas as pd
import data_quality_metric as dqm
import data_preparation.split as sp

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

iris = load_iris()
iris_df = pd.DataFrame(iris.data)
iris_df['class'] = iris.target

X_train, X_test, y_train, y_test = sp.sampling(iris_df, 0.2)

dqm.dq_metric_test(X_train, X_test, y_train, y_test, crt_names, models, 'iris')
# .npy files with the metric will be saved in the output directory
```
####Parallel (test data):
```python
from sklearn.datasets import load_iris
import pandas as pd
import data_quality_metric as dqm
import data_preparation.split as sp
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

iris = load_iris()
iris_df = pd.DataFrame(iris.data)
iris_df['class'] = iris.target

X_train, X_test, y_train, y_test = sp.sampling(iris_df, 0.2)

if __name__ == '__main__':
    freeze_support()
    dqm.dq_metric_test_para(X_train, X_test, y_train, y_test, crt_names, models, 'iris')  # parallel on models
    # .npy files with the metric will be saved in the output directory
```
###Without dedicated test data:
####Not parallel:
```python
from sklearn.datasets import load_iris
import pandas as pd
import data_quality_metric as dqm

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

iris = load_iris()
iris_df = pd.DataFrame(iris.data)
iris_df['class'] = iris.target

dqm.dq_metric(iris_df, crt_names, models, 'iris')  # 30 resamplings by default
# .npy files with the metric will be saved in the output directory
```
####Parallel:
```python
from sklearn.datasets import load_iris
import pandas as pd
import data_quality_metric as dqm
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

iris = load_iris()
iris_df = pd.DataFrame(iris.data)
iris_df['class'] = iris.target

if __name__ == '__main__':
    freeze_support()
    dqm.dq_metric_para(30, iris_df, crt_names, models, 'iris')  # parallel on the 30 resamplings
    # .npy files with the metric will be saved in the output directory
```

##Computing the degree of repairability
We need a version of the dataset before and after repairing, we compute data quality for both 
(or another performance metric) and then use them to compute the degree of repairability. 
In this example we use 1-qa1 as a performance metric.

```python
import pandas as pd
import data_quality_metric as dqm
import repairability as rp
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

#the dataset iris with 25% of a combination of missing values and outliers injected
train_deteriorated = pd.read_csv("dataset/iris/trusted_test/iris_train_missing_outlier_25.csv")
y_train_deteriorated = train_deteriorated["class"].copy()
train_deteriorated.drop(columns=["class"], inplace=True)

#the dataset iris when the errors injected have been repaired with 1. replacing missing values with attributes' means 
# and 2. detecting 10% of outliers with an isolation forest and deleting them
train_repaired = pd.read_csv("dataset/iris/trusted_test/iris_train_m")
y_train_repaired = train_repaired["class"].copy()
train_repaired.drop(columns=["class"], inplace=True)

#trusted test set without errors
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
    perf_deteriorated = 1-qa_det[1] #we use 1-qa1
    _, qa_rep, _, _ = pd.read_csv("output/scores/iris_repaired_example_(x,qa,qf,time).npy")
    perf_repaired = 1-qa_rep[1]
    
    rep_deg = rp.repairability_degree(perf_deteriorated, perf_repaired)
    print(rep_deg)
    
```