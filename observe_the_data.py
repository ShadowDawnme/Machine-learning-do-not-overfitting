import sys, os
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt


train = pd.read_csv('train.csv')
train_y = train['target']
train_X = train.drop(['id', 'target'], axis=1).values

test_df = pd.read_csv('test.csv')
test_df = test_df.drop(['id'], axis=1).values
train.head()
train.info()
train.nunique()

print(train.isnull().sum())
print(train.duplicated().sum())

print(train.mean().sum() / 300)
print(train.std().sum() / 300)

print(test_df.mean().sum() / 300)
print(test_df.std().sum() / 300)

corr_matrix = train.corr()
# Extract the upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
upper

"""
plt.bar(range(2), (train_X.shape[0], test_df.shape[0]), align='center', alpha=0.8)
plt.xticks(range(2), ('train', 'test'))
plt.ylabel('Number of data')
plt.title('The scale comparation')



plt.figure(figsize=(15, 15))
for i in range(10):
    for j in range(10):
        plt.subplot(10, 10, 10 * i + j + 1)
        plt.hist(train[str(10 * i + j)], bins=100)
        plt.title('Column ' + str(10 * i + j))
plt.show()
"""







