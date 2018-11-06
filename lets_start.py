import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv('crx.data', header=None, na_values='?')

print(data.shape)
print(data.head())
print(data.tail())

data.columns = ['A' + str(i) for i in range(1, 16)] + ['class']
print(data.head())


print(data['A5'][687])
print(data.at[687, 'A5'])

print(data.describe())

categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']
print( categorical_columns)
print( numerical_columns)

print(data[categorical_columns].describe())
print(data.describe(include=[object]))

for c in categorical_columns:
        print(data[c].unique())




#from pandas.plotting import scatter_matrix
#print(scatter_matrix(data, alpha=0.05, figsize=(10, 10)));

print(data.corr())


# Prepare data

data = data.fillna(data.median(axis=0), axis=0)
#print(data['A1'].describe())
#data['A1'] = data['A1'].fillna('b')

data_describe = data.describe(include=[object])
for c in categorical_columns:
        data[c] = data[c].fillna(data_describe[c]['top'])
print(data.describe())

binary_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print(binary_columns)
print(nonbinary_columns)

data.at[data['A1'] == 'b', 'A1'] = 0
data.at[data['A1'] == 'a', 'A1'] = 1
print(data['A1'].describe())

data_describe = data.describe(include=[object])
print(data_describe)

for c in binary_columns[1:]:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1

print(data[binary_columns].describe())

print(data['A4'].unique())

data_nonbinary = pd.get_dummies(data[nonbinary_columns])
print(data_nonbinary.columns)

data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
print(data_numerical.describe())

data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)
print (data.shape)
print (data.columns)

X = data.drop(('class'), axis=1)
y = data['class']
feature_names = X.columns
print(feature_names)

print(X.shape)
print(y.shape)
N, d = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)
N_train, _ = X_train.shape
N_test, _ = X_test.shape

print(N_train)
print(N_test)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)

err_train = np.mean(y_train != y_train_predict)
err_test = np.mean(y_test != y_test_predict)
print(err_train)
print(err_test)

from sklearn.model_selection import GridSearchCV
n_neighbors_array = [1, 3, 5, 7, 10, 15]
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid={'n_neighbors' : n_neighbors_array})
grid.fit(X_train, y_train)

best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
print(best_cv_err) 
print(best_n_neighbors) 


knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train, y_train)

err_train = np.mean(y_train != knn.predict(X_train))
err_test  = np.mean(y_test  != knn.predict(X_test))
print(err_train)
print(err_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
print(err_train)
print(err_test)


C_array = np.logspace(-3, 3, num=7)
gamma_array = np.logspace(-5, 2, num=8)
svc = SVC(kernel='rbf')
grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array})
grid.fit(X_train, y_train)
print('CV error    = %f' % (1 - grid.best_score_))
print('best C      = %f' % (grid.best_estimator_.C))
print('best gamma  = %f' % (grid.best_estimator_.gamma))

svc = SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma)
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
print('%f %f' % (err_train, err_test))

C_array = np.logspace(-3, 3, num=7)
svc = SVC(kernel='linear')
grid = GridSearchCV(svc, param_grid={'C': C_array})
grid.fit(X_train, y_train)
print('best error = %f' % (1 - grid.best_score_))
print('best C  = %f' % (grid.best_estimator_.C))

svc = SVC(kernel='linear', C=grid.best_estimator_.C)
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test  = np.mean(y_test  != svc.predict(X_test))
print(err_train)
print(err_test)

from sklearn import ensemble
rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)

err_train = np.mean(y_train != rf.predict(X_train))
err_test  = np.mean(y_test  != rf.predict(X_test))
print(err_train)
print(err_test)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for f, idx in enumerate(indices):
    print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))


best_features = indices[:8]
best_features_names = feature_names[best_features]
print(best_features_names)


gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train, y_train)

err_train = np.mean(y_train != gbt.predict(X_train))
err_test = np.mean(y_test != gbt.predict(X_test))
print(err_train)
print(err_test)


gbt = ensemble.GradientBoostingClassifier(n_estimators=100, random_state=11)
gbt.fit(X_train[best_features_names], y_train)

err_train = np.mean(y_train != gbt.predict(X_train[best_features_names]))
err_test = np.mean(y_test != gbt.predict(X_test[best_features_names]))
print(err_train)
print(err_test)
