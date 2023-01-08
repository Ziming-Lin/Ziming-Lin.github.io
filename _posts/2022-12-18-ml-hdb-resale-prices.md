---
layout: post
title: "Machine Learning with Python - HDB Resale Flat Market"
subtitle: "Deep dive into HDB resale flat prices."
background: '/img/posts/ml-hdb-resale-prices/hdb-landscape.jpg'
---


## Machine Learning Objectives

- To discover patterns in HDB resale flat prices using common measures known to influence prices
- Since the purchase price of individual units are not made publicly available, it is not possible to make ROI the subject of study
- Determine and compare the predictive accuracies of a multiple linear regression model to that of popular XGBoost algorithm model

<br>

## Exploratory Data Analysis

<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/Ziming-Lin/ml-hdb-resale-prices/main/assets/pair_plot.png">  
</p>

<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/Ziming-Lin/ml-hdb-resale-prices/main/assets/displot_resale_date.png">
  <img width="800" src="https://raw.githubusercontent.com/Ziming-Lin/ml-hdb-resale-prices/main/assets/displot_remaining_lease.png">
</p>

<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/Ziming-Lin/ml-hdb-resale-prices/main/assets/heatmap.png"> 
</p>

<ins> EDA Observations </ins>

- COVID-19 severely impacted the market for 2 months and likely had an effect on resale prices
- Focus of study will be post-COVID-19 data
- Spikes in transactions can be observed every 3 months and is possibly due to to the way HDB's resale unit releases
keys to new BTO owners
- It is probable that many owners rush to sell their flats upon reaching the 5-year Minimum Occupation Period (MOP)

<br>


## Multiple Linear Regression Model

With RMSE of 63308 and mean percentage error of 10.82% on the training set, the model coefficients are given as:

<ins> Numerical Features </ins>

<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/Ziming-Lin/ml-hdb-resale-prices/main/assets/numerical_variables.png"> 
</p>

<ins> Categorical Features </ins>

<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/Ziming-Lin/ml-hdb-resale-prices/main/assets/flat_types.png"> 
  <img width="800" src="https://raw.githubusercontent.com/Ziming-Lin/ml-hdb-resale-prices/main/assets/flat_models.png">
</p>

<p align="center">
  <img width="800" src="https://raw.githubusercontent.com/Ziming-Lin/ml-hdb-resale-prices/main/assets/station_names.png"> 
</p>


<ins> Model Interpretation </ins>

- Resale Price = Model Intercept + Categorical Coefficients + Σ(Numerical Coefficient X Standard Scale Numerical Variable)
- Model intercept ($339,246) acts as baseline price
- From the scale of coefficients, station name feature is usually the main price affecting factor
- While it seems that floor area feature is only slightly ahead of remaining lease feature, the size of a flat should be wholly assessed by grouping floor area and flat type
- Feature importance is then:
```
Flat Size > Remaining Lease > Unit Level > Distance to nearest MRT Station > Building Height
```
<br>

## XGBoost Model

Although XGBoost models have a tendency to overfit training datasets, here the XGBoost model is able to perform better than the previous linear regression model.


<ins> Comparison of Test Set Scores </ins>

{:class="table table-bordered"}
| Scoring                   | Multiple Linear Regression | XGBoost (Untuned) | XGBoost (Tuned) |
| --------------------------|----------------------------|-------------------|-----------------|
| Adjusted R-Squared        | 0.8567                     | 0.9314            | 0.9616          |
| Mean Percentage Error (%) | 10.94                      | 7.26              | 4.95            |
| Root Mean Square Error    | 63313                      | 43801             | 32769           |
| Max Error ($)             | 323179                     | 301771            | 275391          |


<ins> Prediction Sample (First 20 in Test Set) </ins>

{:class="table table-bordered"}
| Actual Resale Price       | Predicted Price - <br> Multiple Linear Regression |  Predicted Price - <br> XGBoost (Tuned)|
| :------------------------:|:-------------------------------------------------:|:--------------------------------------:|
| 505000 | 406608 | 518087  |
| 820000 | 702645 | 782638  |
| 525000 | 503319 | 548672  |
| 550000 | 597553 | 574582  |
| 440000 | 394860 | 436342  |
| 470000 | 546282 | 436564  |
| 520000 | 586060 | 553901  |
| 465000 | 483055 | 479597  |
| 280000 | 316829 | 288300  |
| 307000 | 370922 | 309995  |
| 720000 | 652714 | 673095  |
| 395000 | 358667 | 403725  |
| 665000 | 701736 | 710187  |
| 560000 | 644428 | 537112  |
| 440000 | 403256 | 437242  |
| 430000 | 511887 | 422778  |
| 480000 | 450104 | 466556  |
| 468000 | 449832 | 482027  |
| 570000 | 608766 | 595898  |
| 330000 | 316751 | 357157  |


<br>

## Planned Features

- Pre-processing pipeline for newer test data

<br>

You can view the series of Jupyter notebooks in my repository [here](https://github.com/Ziming-Lin/ml-hdb-resale-prices).
