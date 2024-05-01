# Data Quality Metric

## Table of contents

> [Data Quality Evaluation](#Data-Quality-Metric)
> * [Table of contents](#table-of-contents)
> * [Requirements](#requirements)
> * [Examples](#examples)
>    * [With dedicated test data](#with-dedicated-test-data)
>      * [Not parallel (test data)](#not-parallel-test-data)
>      * [Parallel (test data)](#parallel-test-data)
>    * [Without dedicated test data](#without-dedicated-test-data)
>      * [Not parallel](#not-parallel)
>      * [Parallel](#parallel)

## Computing data quality
### Requirements
The classification target needs to be called 'class' in the dataframe, version requirements of packages can be found in 
"requirements.txt"

###Examples:
###With dedicated test data:
####Not parallel (test data):
```python
from sklearn.datasets import load_iris
import pandas as pd
import data_quality_metric as dqm
import split as sp

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
crt_names = ['missing', 'fuzzing', 'outlier']

iris = load_iris()
iris_df = pd.DataFrame(iris.data)
iris_df['class'] = iris.target

X_train, X_test, y_train, y_test = sp.sampling(iris_df, 0.2)

dqm.dq_metric_test(X_train, X_test, y_train, y_test, crt_names, models, 'iris', 'output_name')
# .npy files with the metric will be saved in the output directory
```
####Parallel (test data):
```python
from sklearn.datasets import load_iris
import pandas as pd
import data_quality_metric as dqm
import split as sp
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
    dqm.dq_metric_test_para(X_train, X_test, y_train, y_test, crt_names, models, 'iris', 'output_name')  # parallel on models
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

dqm.dq_metric(iris_df, crt_names, models, 'iris', 'output_name')  # 30 resamplings by default
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
    dqm.dq_metric_para(30, iris_df, crt_names, models, 'iris', 'output_name')  # parallel on the 30 resamplings
    # .npy files with the metric will be saved in the output directory
```
