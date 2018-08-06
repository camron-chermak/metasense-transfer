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
    path = root_dir + "/" + location + "_Round" + str(round) + "/Board" + str(board_id) + "_ReadsFromLog.csv"
    pathEPA = "epaData/epa-" + location + ".csv"
    data = pd.read_csv(path, parse_dates=True) 
    epaData = pd.read_csv(pathEPA, parse_dates=True)
    data['no2'] = data['No2AmV'] - data['No2WmV']
    data['o3']  = data['OxAmV']  - data['OxWmV']
    data['co']  = data['CoAmV']  - data['CoWmV'] 
    data['temperature'] = data['Hum_cT']
    T = data['temperature'] + 273.15
    data['humidity'] = data['Hum_pc']
    data['absolute-humidity'] = data['humidity'] / 100 * np.exp(
        54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T +
        np.tanh(0.0415 * (T - 218.8)) * (53.878 - 1331.22 / T
                                         - 9.44523 * np.log(T) + 0.014025 * T)) / 1000
    data['board'] = board_id
    data['location'] = location
    data['round'] = round  
    data['epa-o3'] = -1
    data['epa-no2'] = -1
  
   
  
    #for index in range(len(epaData)):
        #print(index)
        #print(epaData['datetime'][index])
        #if ( epaData['datetime'][index] == '%d-%b-%Y %H:%M' ):
        #    print('true')
        #epaData.set_value(index, 'abs-time', datetime.datetime.strptime((epaData['datetime'])[index], '%m-%d-% %H:%M'))
        #epaData.set_value(index, 'abs-time', epaData['datetime'][index].strftime('%m-%d-%Y %H:%M'))
    
   # for index in range(len(data)):
    #    data['abs-time'][index] = datetime.datetime.strptime(data['Time'][index], '%m/%d/%Y %I:%M:%S %p')  
  

    #for index in range(len(data)):
    #for index in range(8400):
    #    print(index)  
        #if not pd.isna((epaData[epaData['datetime'] == data['Time'][index]])['epa-o3']).values:  
    #print("start")
   # data = data.reset_index(drop=True)
    #epaData = epaData.reset_index(drop=True)
    #data['epa-o3'] = 1000*(epaData[epaData['datetime'] == data['Time']])['epa-o3'] 
        #if not pd.isna((epaData[epaData['datetime'] == data['Time'][index]])['epa-no2']).values:  
            #data.set_value(index, 'epa-no2', 1000*(epaData[epaData['datetime'] == data['Time'][index]])['epa-no2']) 
       # data.at['epa-no2', index] =  1000*(epaData[epaData['datetime'] == data['Time'][index]])['epa-no2'].values 
    #print("finish")

    epaNo2 = epaData['epa-no2']
    epaO3 = epaData['epa-o3']
    epaTime = epaData['datetime']
    newNo2 = data['epa-no2'].values
    newNo2 = np.expand_dims(newNo2, axis=1) 
    newO3 = data['epa-o3'].values
    newO3 = np.expand_dims(newO3, axis=1) 
    
    for index in range(len(data)):
        print(index)
        if not np.isnan(epaNo2[np.where(epaTime == data['Time'][index])[0]]).values: 
            newNo2[index] = 1000*epaNo2[np.where(epaTime == data['Time'][index])[0]]    
        if not np.isnan(epaO3[np.where(epaTime == data['Time'][index])[0]]).values: 
            newO3[index] = 1000*epaO3[np.where(epaTime == data['Time'][index])[0]]     
        #print(epaTime.index(data['Time'][index]))
    data['epa-no2'] = newNo2
    data['epa-o3'] = newO3
    data = data[data['epa-no2'] != -1]
    data = data[data['epa-o3'] != -1]
    data.to_csv('don.csv') 
    train, test = train_test_split(data, test_size=0.2, random_state=seed)
    return train, test, epaData 
