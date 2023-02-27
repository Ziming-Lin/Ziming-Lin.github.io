---
layout: post
title: "Machine Learning with Python - Kaggle Competition (Stroke Prediction)"
subtitle: "Dealing with Stroke Prediction Classification Dataset"
background: '/img/posts/ml-kaggle-stroke/stroke banner image.jpg'
---


## Introduction

- The dataset is from a Kaggle competition [<ins>here</ins>](https://www.kaggle.com/competitions/playground-series-s3e2/overview).
- Estimator used here is XGBoost with prediction accuracy of only 54% as the model still needs to have its hyperparameters tuned (in the works)

<br>

## Exploratory Data Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
```


```python
df = pd.read_csv('train.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Male</td>
      <td>28.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>79.53</td>
      <td>31.1</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>78.44</td>
      <td>23.9</td>
      <td>formerly smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Female</td>
      <td>42.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>103.00</td>
      <td>40.3</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Male</td>
      <td>56.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>64.87</td>
      <td>28.8</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Female</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>Private</td>
      <td>Rural</td>
      <td>73.36</td>
      <td>28.8</td>
      <td>never smoked</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_test = pd.read_csv('test.csv')
df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15304</td>
      <td>Female</td>
      <td>57.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>82.54</td>
      <td>33.4</td>
      <td>Unknown</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15305</td>
      <td>Male</td>
      <td>70.0</td>
      <td>1</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>72.06</td>
      <td>28.5</td>
      <td>Unknown</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15306</td>
      <td>Female</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Urban</td>
      <td>103.72</td>
      <td>19.5</td>
      <td>Unknown</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15307</td>
      <td>Female</td>
      <td>56.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Govt_job</td>
      <td>Urban</td>
      <td>69.24</td>
      <td>41.4</td>
      <td>smokes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15308</td>
      <td>Male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>111.15</td>
      <td>30.1</td>
      <td>smokes</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15304.000000</td>
      <td>15304.000000</td>
      <td>15304.000000</td>
      <td>15304.000000</td>
      <td>15304.000000</td>
      <td>15304.000000</td>
      <td>15304.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7651.500000</td>
      <td>41.417708</td>
      <td>0.049726</td>
      <td>0.023327</td>
      <td>89.039853</td>
      <td>28.112721</td>
      <td>0.041296</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4418.028595</td>
      <td>21.444673</td>
      <td>0.217384</td>
      <td>0.150946</td>
      <td>25.476102</td>
      <td>6.722315</td>
      <td>0.198981</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>55.220000</td>
      <td>10.300000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3825.750000</td>
      <td>26.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>74.900000</td>
      <td>23.500000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7651.500000</td>
      <td>43.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>85.120000</td>
      <td>27.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11477.250000</td>
      <td>57.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>96.980000</td>
      <td>32.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15303.000000</td>
      <td>82.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>267.600000</td>
      <td>80.100000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    id                   0
    gender               0
    age                  0
    hypertension         0
    heart_disease        0
    ever_married         0
    work_type            0
    Residence_type       0
    avg_glucose_level    0
    bmi                  0
    smoking_status       0
    stroke               0
    dtype: int64




```python
df.select_dtypes('object')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>smoking_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>never smoked</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Male</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>formerly smoked</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>Unknown</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Male</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>never smoked</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Female</td>
      <td>No</td>
      <td>Private</td>
      <td>Rural</td>
      <td>never smoked</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15299</th>
      <td>Female</td>
      <td>No</td>
      <td>Govt_job</td>
      <td>Urban</td>
      <td>never smoked</td>
    </tr>
    <tr>
      <th>15300</th>
      <td>Female</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>never smoked</td>
    </tr>
    <tr>
      <th>15301</th>
      <td>Female</td>
      <td>Yes</td>
      <td>Self-employed</td>
      <td>Urban</td>
      <td>never smoked</td>
    </tr>
    <tr>
      <th>15302</th>
      <td>Male</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Rural</td>
      <td>Unknown</td>
    </tr>
    <tr>
      <th>15303</th>
      <td>Female</td>
      <td>No</td>
      <td>Private</td>
      <td>Rural</td>
      <td>never smoked</td>
    </tr>
  </tbody>
</table>
<p>15304 rows × 5 columns</p>
</div>




```python
df.dtypes
```




    id                     int64
    gender                object
    age                  float64
    hypertension           int64
    heart_disease          int64
    ever_married          object
    work_type             object
    Residence_type        object
    avg_glucose_level    float64
    bmi                  float64
    smoking_status        object
    stroke                 int64
    dtype: object




```python
df.nunique()
```




    id                   15304
    gender                   3
    age                    106
    hypertension             2
    heart_disease            2
    ever_married             2
    work_type                5
    Residence_type           2
    avg_glucose_level     3740
    bmi                    407
    smoking_status           4
    stroke                   2
    dtype: int64




```python
# Look into oddities in gender, age, smoking statuss

df.gender.value_counts()
```




    Female    9446
    Male      5857
    Other        1
    Name: gender, dtype: int64




```python
# Much more females than males. Check 'Other'.
df[df.gender == 'Other']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9363</th>
      <td>9363</td>
      <td>Other</td>
      <td>9.0</td>
      <td>0</td>
      <td>0</td>
      <td>No</td>
      <td>children</td>
      <td>Rural</td>
      <td>96.04</td>
      <td>18.0</td>
      <td>Unknown</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for 'Other' gender in test set too.
df_test[df_test.gender == 'Other']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>gender</th>
      <th>age</th>
      <th>hypertension</th>
      <th>heart_disease</th>
      <th>ever_married</th>
      <th>work_type</th>
      <th>Residence_type</th>
      <th>avg_glucose_level</th>
      <th>bmi</th>
      <th>smoking_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4629</th>
      <td>19933</td>
      <td>Other</td>
      <td>56.0</td>
      <td>0</td>
      <td>0</td>
      <td>Yes</td>
      <td>Private</td>
      <td>Urban</td>
      <td>102.53</td>
      <td>35.0</td>
      <td>Unknown</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Since it exist in both sets, we will not drop it.
```


```python
pd.options.display.max_rows = 200
for col in df[['age', 'smoking_status']]:
    print('---' + col + '---')
    print(df[col].value_counts())
    print('\n')
```

    ---age---
    57.00    353
    78.00    337
    53.00    311
    31.00    310
    45.00    309
    52.00    308
    55.00    303
    37.00    295
    50.00    294
    56.00    286
    54.00    285
    49.00    282
    32.00    275
    43.00    273
    5.00     270
    79.00    269
    51.00    265
    34.00    264
    39.00    263
    44.00    256
    61.00    254
    59.00    248
    47.00    247
    42.00    246
    38.00    243
    40.00    236
    60.00    230
    41.00    230
    26.00    227
    62.00    224
    58.00    218
    27.00    210
    8.00     206
    46.00    203
    48.00    200
    63.00    195
    18.00    194
    20.00    191
    23.00    188
    17.00    188
    33.00    182
    80.00    180
    14.00    176
    13.00    173
    2.00     172
    30.00    170
    25.00    170
    24.00    167
    35.00    165
    16.00    165
    65.00    161
    28.00    154
    3.00     146
    66.00    144
    19.00    130
    81.00    127
    29.00    125
    21.00    119
    12.00    109
    36.00    108
    64.00    104
    71.00     98
    22.00     98
    69.00     96
    82.00     94
    15.00     91
    67.00     91
    10.00     85
    4.00      84
    75.00     84
    9.00      84
    7.00      78
    73.00     74
    11.00     73
    76.00     71
    72.00     68
    74.00     62
    70.00     61
    68.00     60
    77.00     57
    6.00      49
    1.88      43
    1.32      37
    1.80      36
    1.08      31
    1.24      30
    1.64      24
    1.72      24
    0.72      22
    1.48      18
    0.56      16
    0.80      16
    0.32      15
    0.88      12
    1.00      12
    1.16      12
    1.56      11
    1.40      10
    0.24      10
    0.64       9
    0.40       7
    0.16       6
    0.08       6
    0.48       3
    1.30       2
    0.68       1
    Name: age, dtype: int64
    
    
    ---smoking_status---
    never smoked       6281
    Unknown            4543
    formerly smoked    2337
    smokes             2143
    Name: smoking_status, dtype: int64
    
    
    


```python
# Getting to know other features of the dataset

for col in df[['hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'stroke']]:
    print('---' + col + '---')
    print(df[col].value_counts())
    print('\n')
```

    ---hypertension---
    0    14543
    1      761
    Name: hypertension, dtype: int64
    
    
    ---heart_disease---
    0    14947
    1      357
    Name: heart_disease, dtype: int64
    
    
    ---ever_married---
    Yes    10385
    No      4919
    Name: ever_married, dtype: int64
    
    
    ---work_type---
    Private          9752
    children         2038
    Self-employed    1939
    Govt_job         1533
    Never_worked       42
    Name: work_type, dtype: int64
    
    
    ---Residence_type---
    Rural    7664
    Urban    7640
    Name: Residence_type, dtype: int64
    
    
    ---stroke---
    0    14672
    1      632
    Name: stroke, dtype: int64
    
    
    
<br>

## Graphical EDA


```python
numerical = [
    'age', 'avg_glucose_level', 'bmi'
]
categorical = [
  'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease', 'stroke'
]

dataset = df[numerical + categorical]
dataset.shape
```




    (15304, 11)




```python
dataset[numerical].hist(figsize=(20, 10), layout=(2, 4));
```

<p align="center">
  <img src="/img/posts/ml-kaggle-stroke/output_17_0.png">  
</p>


```python
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for variable, subplot in zip(categorical, ax.flatten()):
    sns.countplot(data=dataset, x=variable, ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(20)
```

<p align="center">
  <img src="/img/posts/ml-kaggle-stroke/output_18_0.png">  
</p>
       

### Stroke Analysis


```python
sns.pairplot(dataset, vars=['age', 'avg_glucose_level', 'bmi'], hue = 'stroke')
plt.show()
```

<p align="center">
  <img src="/img/posts/ml-kaggle-stroke/output_20_0.png">  
</p>




```python
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
for variable, subplot in zip(['hypertension', 'heart_disease'], ax.flatten()):
    sns.kdeplot(dataset[dataset["stroke"] == 1][variable], label="Stroke", shade=True, color="r", ax = subplot)
    sns.kdeplot(dataset[dataset["stroke"] == 0][variable], label="No stroke", shade=True, color="g", ax = subplot)
```

<p align="center">
  <img src="/img/posts/ml-kaggle-stroke/output_21_0.png">  
</p>

    

# XGBoost


```python
# Pipeline prep

df = pd.read_csv('train.csv')
df = df.dropna()

df_test = pd.read_csv('test.csv')

X_train = df.drop(['id','stroke'], axis=1)
y_train = df.stroke

X_test = df_test.drop('id', axis=1)
```


```python
# To use StandardScaler on:
# age, avg_glucose_level, bmi

# To use OHE on:
# gender, work_type, Residence_type

# To use OE on:
# ever_married, smoking_status

# Passthrough:
# hypertension, heart_disease
```


```python
ohe_columns = ['gender', 'work_type', 'Residence_type']
ord_columns = ['ever_married', 'smoking_status']
num_columns = ['age', 'avg_glucose_level', 'bmi']
passthrough_columns = ['hypertension', 'heart_disease']
```


```python
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
```


```python
ohe = OneHotEncoder(categories=
                    [['Male', 'Female', 'Other'],
                     ['Private', 'children', 'Self-employed', 'Govt_job', 'Never_worked'],
                     ['Urban', 'Rural']])
ordinal = OrdinalEncoder(categories=
                         [['No', 'Yes'],
                          ['never smoked', 'Unknown', 'formerly smoked', 'smokes']])
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', ohe, ohe_columns),
        ('ord', ordinal, ord_columns),
        ('std', scaler, num_columns),
        ('passthrough', 'passthrough', passthrough_columns)],
    remainder='drop'
)
```


```python
X_train_transformed = preprocessor.fit_transform(X_train)
X_train_transformed
```




    array([[ 1.        ,  0.        ,  0.        , ...,  0.44439699,
             0.        ,  0.        ],
           [ 1.        ,  0.        ,  0.        , ..., -0.62669753,
             0.        ,  0.        ],
           [ 0.        ,  1.        ,  0.        , ...,  1.81301777,
             0.        ,  0.        ],
           ...,
           [ 0.        ,  1.        ,  0.        , ..., -0.28454234,
             0.        ,  0.        ],
           [ 1.        ,  0.        ,  0.        , ..., -0.83496591,
             0.        ,  0.        ],
           [ 0.        ,  1.        ,  0.        , ..., -0.50768703,
             0.        ,  0.        ]])




```python
from xgboost import XGBClassifier

XGBclf = XGBClassifier(objective='binary:logistic')
XGBclf.fit(X_train_transformed, y_train)
```




    XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                  early_stopping_rounds=None, enable_categorical=False,
                  eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
                  grow_policy='depthwise', importance_type=None,
                  interaction_constraints='', learning_rate=0.300000012,
                  max_bin=256, max_cat_threshold=64, max_cat_to_onehot=4,
                  max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                  missing=nan, monotone_constraints='()', n_estimators=100,
                  n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0, ...)




```python
X_test_transformed = preprocessor.transform(X_test)

prediction = XGBclf.predict(X_test_transformed)
```


```python
print(prediction)
print(np.unique(prediction))
df_results = pd.DataFrame(prediction)
```

    [0 0 0 ... 0 0 0]
    [0 1]
    


```python
df_results[0].sum()
```




    131




```python
len(prediction)
```




    10204



# Create Pipeline


```python
# Create ML pipeline

ml_pipeline = Pipeline([('preprocessing', preprocessor), ('xgbclf', XGBClassifier(objective='binary:logistic'))])
```


```python
ml_pipeline.fit(X_train, y_train)
```




    Pipeline(steps=[('preprocessing',
                     ColumnTransformer(transformers=[('ohe',
                                                      OneHotEncoder(categories=[['Male',
                                                                                 'Female',
                                                                                 'Other'],
                                                                                ['Private',
                                                                                 'children',
                                                                                 'Self-employed',
                                                                                 'Govt_job',
                                                                                 'Never_worked'],
                                                                                ['Urban',
                                                                                 'Rural']]),
                                                      ['gender', 'work_type',
                                                       'Residence_type']),
                                                     ('ord',
                                                      OrdinalEncoder(categories=[['No',
                                                                                  'Yes'],
                                                                                 ['never '
                                                                                  'smoked',
                                                                                  'Unknown',
                                                                                  'formerly '
                                                                                  'smoked',
                                                                                  'smokes...
                                   feature_types=None, gamma=0, gpu_id=-1,
                                   grow_policy='depthwise', importance_type=None,
                                   interaction_constraints='',
                                   learning_rate=0.300000012, max_bin=256,
                                   max_cat_threshold=64, max_cat_to_onehot=4,
                                   max_delta_step=0, max_depth=6, max_leaves=0,
                                   min_child_weight=1, missing=nan,
                                   monotone_constraints='()', n_estimators=100,
                                   n_jobs=0, num_parallel_tree=1, predictor='auto',
                                   random_state=0, ...))])




```python
ml_prediction = ml_pipeline.predict(X_test)
```


```python
len(ml_prediction)
```




    10204




```python
print(ml_prediction)
print(np.unique(ml_prediction))
print(np.unique(prediction == ml_prediction))
```

    [0 0 0 ... 0 0 0]
    [0 1]
    [ True]
    


```python
output = pd.DataFrame({'id': df_test.id, 'stroke': ml_prediction})
output
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>stroke</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15304</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15305</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15306</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15307</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15308</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10199</th>
      <td>25503</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10200</th>
      <td>25504</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10201</th>
      <td>25505</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10202</th>
      <td>25506</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10203</th>
      <td>25507</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10204 rows × 2 columns</p>
</div>



# Untuned Result Submission


```python
output.to_csv('submission.csv', index=False)
```
