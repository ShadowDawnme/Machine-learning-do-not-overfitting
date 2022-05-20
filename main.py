import sys, os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import ExtraTreesClassifier

seed = 210
np.random.seed(seed)
train = pd.read_csv('train.csv')
train_y = train['target']
train_X = train.drop(['id','target'], axis=1).values
test_df = pd.read_csv('test.csv')
test_df= test_df.drop(['id'], axis=1).values
train.head()
train.info()
train.nunique()


data = RobustScaler().fit_transform(np.concatenate((train_X, test_df), axis=0))
train_X = data[:250]
test_df= data[250:]
noise = 0.01
train_X += np.random.normal(0, noise, train_X.shape)

train.drop(columns=['id','target']).corr()

HistGradient = ExtraTreesClassifier(warm_start=True)

param = {'n_estimators': [250, 500],
         'max_depth': [11],
         'min_samples_split': [9],
         'min_samples_leaf': [9],
         }

def scoring_roc_auc(y, y_pred):
    try:
        return roc_auc_score(y, y_pred)
    except:
        return 0.5

robust_roc_auc = make_scorer(scoring_roc_auc)

gridSearch_HistGradient = GridSearchCV(HistGradient, param, scoring=robust_roc_auc, cv=7, verbose=3)
gridSearch_HistGradient.fit(train_X, train_y)
best_HistGradient = gridSearch_HistGradient.best_estimator_
bestHistGradient_testScore = best_HistGradient.score(train_X, train_y)
cv = 21
model = Lasso(alpha=0.031, tol=0.01, warm_start=True, random_state=seed, selection='random')
param_grid = {'alpha' : [0.022, 0.021, 0.02, 0.019, 0.023, 0.024, 0.025, 0.026, 0.027, 0.029, 0.031],
              'tol': [0.0013, 0.0014, 0.001, 0.0015, 0.0011, 0.0012, 0.0016, 0.0017]}
min_features = 12
the_step = 16
feature_selector = RFECV(model, min_features_to_select=min_features, scoring=robust_roc_auc, step=the_step, verbose=0, cv=cv, n_jobs=-1)

predictions = pd.DataFrame()
counter = 0
grid = 21
test_size = 0.4
splits = 21

for train_index, val_index in StratifiedShuffleSplit(n_splits=splits, test_size=test_size, random_state=seed).split(train_X, train_y):
    X, val_X = train_X[train_index], train_X[val_index]
    y, val_y = train_y[train_index], train_y[val_index]
    feature_selector.fit(X, y)

    X_important_features = feature_selector.transform(X)
    val_X_important_features = feature_selector.transform(val_X)
    test_important_features = feature_selector.transform(test_df)

    grid_search = GridSearchCV(feature_selector.estimator_, param_grid=param_grid, verbose=0, n_jobs=-1,
                               scoring=robust_roc_auc, cv=20)
    grid_search.fit(X_important_features, y)

    val_y_pred = grid_search.best_estimator_.predict(val_X_important_features)
    val_mse = mean_squared_error(val_y, val_y_pred)
    val_mae = mean_absolute_error(val_y, val_y_pred)
    val_roc = roc_auc_score(val_y, val_y_pred)
    val_cos = cosine_similarity(val_y.values.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_dst = euclidean_distances(val_y.values.reshape(1, -1), val_y_pred.reshape(1, -1))[0][0]
    val_r2 = r2_score(val_y, val_y_pred)
    shold = 0.2
    if val_r2 > shold:
        message = '<-- OK'
        prediction = grid_search.best_estimator_.predict(test_important_features)
        predictions = pd.concat([predictions, pd.DataFrame(prediction)], axis=1)
    else:
        message = '<-- skipping'
    counter += 1

mean_pred = pd.DataFrame(predictions.mean(axis=1))
mean_pred.index += 250
mean_pred.columns = ['target']
mean_pred.to_csv('submission.csv', index_label='id', index=True)
