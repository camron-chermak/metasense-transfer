import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import datetime

def match5Second(round, location, board_id, root_dir='Data5Second', seed=0):
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
  
    epaNo2 = epaData['epa-no2']
    epaO3 = epaData['epa-o3']
    epaTime = epaData['datetime']
    newNo2 = data['epa-no2'].values
    newNo2 = np.expand_dims(newNo2, axis=1) 
    newO3 = data['epa-o3'].values
    newO3 = np.expand_dims(newO3, axis=1) 
    
    perDone = 0 
    for index in range(len(data)):
        prev = perDone
        print((index / len(data)) * 100)  
        if len(np.where(epaTime == data['Time'][index])) > 0 and len(np.where(epaTime == data['Time'][index])[0]) > 0: 
            if not np.isnan(epaNo2[np.where(epaTime == data['Time'][index])[0]]).values[0]:  
                newNo2[index] = 1000*epaNo2[(np.where(epaTime == data['Time'][index])[0])[0]]    
            if not np.isnan(epaO3[np.where(epaTime == data['Time'][index])[0]]).values[0]: 
                newO3[index] = 1000*epaO3[(np.where(epaTime == data['Time'][index])[0])[0]]      
    data['epa-no2'] = newNo2
    data['epa-o3'] = newO3
    data = data[data['epa-no2'] != -1]
    data = data[data['epa-o3'] != -1]
    outPath = "match_round_" + str(round) + "_" + location + "_" + str(board_id) + ".csv" 
    data.to_csv(outPath) 
    train, test = train_test_split(data, test_size=0.2, random_state=seed)
    return train, test 



#data = match5Second(1, 'donovan', 17)
#data = match5Second(1, 'donovan', 19)
#data = match5Second(1, 'donovan', 21) 
#data = match5Second(2, 'donovan', 15)
#data = match5Second(2, 'donovan', 18)
#data = match5Second(2, 'donovan', 20) #TODO
#data = match5Second(3, 'donovan', 11) #TODO
#data = match5Second(3, 'donovan', 12) #TODO
#data = match5Second(3, 'donovan', 13) #TODO
#data = match5Second(4, 'donovan', 17) #TODO
#data = match5Second(4, 'donovan', 19) #TODO
#data = match5Second(4, 'donovan', 21) #TODO

