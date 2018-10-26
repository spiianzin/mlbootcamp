import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
