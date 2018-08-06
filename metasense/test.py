import metasense
import data
from data import loadMatrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

data = loadMatrix( 1, 'donovan', 19)
train = data[0]
train = train.sort_values(by=['datetime'])
mat = train['datetime'].values
mat = np.expand_dims(mat, axis=1)
mat = np.concatenate( (mat, np.expand_dims(train['no2'].values, axis=1)), axis=1)
mat = np.concatenate( (mat, np.expand_dims(train['o3'].values, axis=1)), axis=1)
mat = np.concatenate( (mat, np.expand_dims(train['co'].values, axis=1)), axis=1)
mat = np.concatenate( (mat, np.expand_dims(train['epa-no2'].values, axis=1)), axis=1)
mat = np.concatenate( (mat, np.expand_dims(train['epa-o3'].values, axis=1)), axis=1)
mat = np.concatenate( (mat, np.expand_dims(train['temperature'].values, axis=1)), axis=1)
mat = np.concatenate( (mat, np.expand_dims(train['absolute-humidity'].values, axis=1)), axis=1)
tempDates = np.chararray((len(train['datetime']), 1), itemsize=30)
for index in range(len(tempDates)):
    if (index - 1) >= 0:
        tempDates[index] = train['datetime'].values[index - 1]
    else:
        tempDates[index] = "no data"
mat = np.concatenate( (mat, tempDates), axis=1)
temp = np.zeros(shape=(len(train['datetime']), 1))
for index in range(len(temp)):
    if (index - 1) >= 0:
        temp[index] = train['no2'].values[index - 1]
mat = np.concatenate( (mat, temp), axis=1)
temp = np.zeros(shape=(len(train['datetime']), 1))
for index in range(len(temp)):
    if (index - 1) >= 0:
        temp[index] = train['o3'].values[index - 1]
mat = np.concatenate( (mat, temp), axis=1)
temp = np.zeros(shape=(len(train['datetime']), 1))
for index in range(len(temp)):
    if (index - 1) >= 0:
        temp[index] = train['co'].values[index - 1]
mat = np.concatenate( (mat, temp), axis=1)
temp = np.zeros(shape=(len(train['datetime']), 1))
for index in range(len(temp)):
    if (index - 1) >= 0:
        temp[index] = train['epa-no2'].values[index - 1]
mat = np.concatenate( (mat, temp), axis=1)
temp = np.zeros(shape=(len(train['datetime']), 1))
for index in range(len(temp)):
    if (index - 1) >= 0:
        temp[index] = train['epa-o3'].values[index - 1]
mat = np.concatenate( (mat, temp), axis=1)
temp = np.zeros(shape=(len(train['datetime']), 1))
for index in range(len(temp)):
    if (index - 1) >= 0:
        temp[index] = train['temperature'].values[index - 1]
mat = np.concatenate( (mat, temp), axis=1)
temp = np.zeros(shape=(len(train['datetime']), 1))
for index in range(len(temp)):
    if (index - 1) >= 0:
        temp[index] = train['absolute-humidity'].values[index - 1]
mat = np.concatenate( (mat, temp), axis=1)
