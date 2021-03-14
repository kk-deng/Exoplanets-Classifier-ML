# Machine-Learning-Challenge

<img src="https://github.com/kk-deng/Machine-Learning-Challenge/blob/main/image/nasa-incredible-map-4000-exoplanets.jpg">

Over a period of nine years in deep space, the NASA Kepler space telescope has been out on a planet-hunting mission to discover hidden planets outside of our solar system.
To help process this data, you will create machine learning models capable of classifying candidate exoplanets from the raw dataset.

## Files Index

Following files are attached:

1. <a href="https://github.com/kk-deng/Machine-Learning-Challenge/blob/main/model_1.ipynb">model_1.ipynb</a>: Model 1 with KNN classifier

2. <a href="https://github.com/kk-deng/Machine-Learning-Challenge/blob/main/model_2.ipynb">model_2.ipynb</a>: Model 2 with Logistic Regression

3. <a href="https://github.com/kk-deng/Machine-Learning-Challenge/blob/main/model_3.ipynb">model_2.ipynb</a>: Model 3 with Random Forest

4. <a href="https://github.com/kk-deng/Machine-Learning-Challenge/blob/main/model_rf.sav">model_rf.sav</a>: Dumped trained model file

### GridSearch for Optimization of Model Parameters

* For KNN model:

```python
param_grid = {
    "n_neighbors": range(1, 20, 2),
    "weights": ['uniform', 'distance'],
    "metric": ["euclidean", "manhattan"]
}

# Output of Best Estimator: 
KNeighborsClassifier(metric='manhattan', n_neighbors=19, weights='distance')

```

* For Logistic Regression model:

```python
param_grid = {
    "C": np.logspace(-3,3,7),
    "penalty": ['l1', 'l2'],
    "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag']
}

# Output of Best Estimator: 
LogisticRegression(C=10.0, penalty="l2", solver='newton-cg')

```

* For Random Forest model:

```python
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

# Output of Best Estimator: 
RandomForestClassifier(criterion='entropy', max_depth=8, max_features='log2', n_estimators=200)

```

### Tesing Score Comparison

| Model | Testing Data Score |
|---|---|
| KNN | 64.8% |
| Logistic Regression | 61.5% |
| Random Forest | 74.5% |

By using the GridSearchCV function, it takes a dict of all possible parameters in a for loop for getting the best parameters. If `n_jobs=-1` is given, the search can be done in parallel calculation which saves a lot of time in fitting hundreds of candidates/fits.

So far, Random Forest model has the best score among these three models. Results of other models are to be determined. The current model is still not good enough to predict new exoplanets with only 75% of accuracy. 
