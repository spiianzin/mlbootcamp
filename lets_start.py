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
