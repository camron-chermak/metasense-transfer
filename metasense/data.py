import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import datetime

def load(round, location, board_id, root_dir=Path('data/final'), seed=0):
    path = root_dir / ("round%u" % round) / location / str("%u.csv" % board_id)
    data = pd.read_csv(path, parse_dates=True, index_col='datetime')
    data['no2'] = data['no2-A'] - data['no2-W']
    data['o3']  = data['o3-A']  - data['o3-W']
    data['co']  = data['co-A']  - data['co-W']
    data['epa-no2'] *= 1000
    data['epa-o3'] *= 1000
    T = data['temperature'] + 273.15
    data['absolute-humidity'] = data['humidity'] / 100 * np.exp(
        54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T +
        np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T
                                         - 9.44523 * np.log(T) + 0.014025 * T)) / 1000
    data['board'] = board_id
    data['location'] = location
    data['round'] = round
    train, test = train_test_split(data, test_size=0.2, random_state=seed)
    return train, test

def loadMatrix(timeStep, round, location, board_id, root_dir=Path('data/final'), seed=0):   
    path = root_dir / ("round%u" % round) / location / str("%u.csv" % board_id)
    data = pd.read_csv(path, parse_dates=True) 
    data['no2'] = data['no2-A'] - data['no2-W'] 
    data['o3']  = data['o3-A'] - data['o3-W']
    data['co']  = data['co-A'] - data['co-W']
    data['epa-no2'] *= 1000
    data['epa-o3'] *= 1000
    T = data['temperature'] + 273.15
    data['absolute-humidity'] = data['humidity'] / 100 * np.exp(
        54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T +
        np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T
                                         - 9.44523 * np.log(T) + 0.014025 * T)) / 1000
    data['board'] = board_id
    data['location'] = location
    data['round'] = round
    train, test = train_test_split(data, test_size=0.2, random_state=seed)
    
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
    for time in range(timeStep):
        timeStep = time + 1
        tempDates = np.chararray((len(train['datetime']), 1), itemsize=30)
        for index in range(len(tempDates)):
            if (index - timeStep) >= 0:
                tempDates[index] = train['datetime'].values[index - timeStep]
            else:
                tempDates[index] = "no data"
        mat = np.concatenate( (mat, tempDates), axis=1)
        temp = np.zeros(shape=(len(train['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['no2'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['o3'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['co'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['epa-no2'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['epa-o3'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['temperature'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['absolute-humidity'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['datetime']), 1))
    train = mat
   
    test = test.sort_values(by=['datetime'])
    mat = test['datetime'].values
    mat = np.expand_dims(mat, axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['no2'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['o3'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['co'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['epa-no2'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['epa-o3'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['temperature'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['absolute-humidity'].values, axis=1)), axis=1)
    for time in range(timeStep):
        timeStep = time + 1
        tempDates = np.chararray((len(test['datetime']), 1), itemsize=30)
        for index in range(len(tempDates)):
            if (index - timeStep) >= 0:
                tempDates[index] = test['datetime'].values[index - timeStep]
            else:
                tempDates[index] = "no data"
        mat = np.concatenate( (mat, tempDates), axis=1)
        temp = np.zeros(shape=(len(test['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['no2'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['o3'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['co'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['epa-no2'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['epa-o3'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['temperature'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['datetime']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['absolute-humidity'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['datetime']), 1))

    test = mat
    
    return train, test


def load5Second(round, location, board_id, root_dir='Data5Second', seed=0):

    path = "match_5Second/" + location + "/Round" + str(round) + "/match_round_" + str(round) + "_" + location + "_" + str(board_id) + ".csv"
    data = pd.read_csv(path, parse_dates=True)   

    train, test = train_test_split(data, test_size=0.2, random_state=seed)
 
    return train, test 
