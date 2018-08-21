import metasense
import metasense.data
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sklearn
#from sklearn import linear_model
#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

NO2 = 0
O3 = 1
PREDICT = 0
ACTUAL = 1

def load3Point( round, location, board_id, seed=0):   
      
    path = "match_5Second/" + location + "/Round" + str(round) + "/match_round_" + str(round) + "_" + location + "_" + str(board_id) + ".csv"
    data = pd.read_csv(path, parse_dates=True)   


    
    data = data.sort_values(by=['Time'])
    mat = data['Time'].values
    mat = np.expand_dims(mat, axis=1)
    mat = np.concatenate( (mat, np.expand_dims(data['no2'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(data['o3'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(data['co'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(data['epa-no2'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(data['epa-o3'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(data['temperature'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(data['absolute-humidity'].values, axis=1)), axis=1)

    # One Back
    tempDates = data['Time'].values
    tempDates = np.expand_dims(tempDates, axis=1)
    mat = np.concatenate( (mat, tempDates), axis=1)

    temp1 = data['no2'].values 
    temp1 = np.pad( temp1, (1, 0), 'constant' ) 
    temp1 = temp1[ : -1 ]
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1) 
      
    temp1 = data['o3'].values 
    temp1 = np.pad( temp1, (1, 0), 'constant' ) 
    temp1 = temp1[ : -1 ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)

    temp1 = data['co'].values 
    temp1 = np.pad( temp1, (1, 0), 'constant' ) 
    temp1 = temp1[ : -1 ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)

    temp1 = data['epa-no2'].values 
    temp1 = np.pad( temp1, (1, 0), 'constant' ) 
    temp1 = temp1[ : -1 ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)

    temp1 = data['epa-o3'].values 
    temp1 = np.pad( temp1, (1, 0), 'constant' ) 
    temp1 = temp1[ : -1 ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)
 
    temp1 = data['temperature'].values 
    temp1 = np.pad( temp1, (1, 0), 'constant' ) 
    temp1 = temp1[ : -1 ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)
 
    temp1 = data['absolute-humidity'].values 
    temp1 = np.pad( temp1, (1, 0), 'constant' ) 
    temp1 = temp1[ : -1 ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)
       

    # One Forward
    tempDates = data['Time'].values
    tempDates = np.expand_dims(tempDates, axis=1)
    mat = np.concatenate( (mat, tempDates), axis=1)

    temp1 = data['no2'].values 
    temp1 = np.pad( temp1, (0, 1), 'constant' ) 
    temp1 = temp1[ 1 : ]
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1) 
      
    temp1 = data['o3'].values 
    temp1 = np.pad( temp1, (0, 1), 'constant' ) 
    temp1 = temp1[ 1 : ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)

    temp1 = data['co'].values 
    temp1 = np.pad( temp1, (0, 1), 'constant' ) 
    temp1 = temp1[ 1 : ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)

    temp1 = data['epa-no2'].values 
    temp1 = np.pad( temp1, (0, 1), 'constant' ) 
    temp1 = temp1[1 : ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)

    temp1 = data['epa-o3'].values 
    temp1 = np.pad( temp1, (0, 1), 'constant' ) 
    temp1 = temp1[1: ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)
 
    temp1 = data['temperature'].values 
    temp1 = np.pad( temp1, (0, 1), 'constant' ) 
    temp1 = temp1[ 1 : ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)
 
    temp1 = data['absolute-humidity'].values 
    temp1 = np.pad( temp1, (0, 1), 'constant' ) 
    temp1 = temp1[ 1 : ]  
    temp1 = np.expand_dims(temp1, axis=1)
    mat = np.concatenate( (mat, temp1), axis=1)
       
    mat = mat[1: ] 
    mat = mat[0: -1]

    mat[:, 1] = (mat[:, 1] + mat[:, 9] + mat[:, 17]) / 3
    mat[:, 2] = (mat[:, 2] + mat[:, 10] + mat[:, 18]) / 3
    mat[:, 3] = (mat[:, 3] + mat[:, 11] + mat[:, 19]) / 3
    mat[:, 4] = (mat[:, 4] + mat[:, 12] + mat[:, 20]) / 3
    mat[:, 5] = (mat[:, 5] + mat[:, 13] + mat[:, 21]) / 3
    mat[:, 6] = (mat[:, 6] + mat[:, 14] + mat[:, 22]) / 3
    mat[:, 7] = (mat[:, 7] + mat[:, 15] + mat[:, 23]) / 3
    mat = mat[ :, 1 : 8]

    df = pd.DataFrame( mat )

    train, test = train_test_split(df, test_size=0.2, random_state=seed)   
    train = train.values
    test = test.values 
    return train, test


def train3Point( round, location, board ):
    rf = RandomForestRegressor(random_state=0)
    data = load3Point( round, location, board)
    train = data[0]
    test = data[1]

    trainX = train[:, 0:3]
    trainX = np.concatenate( (trainX, train[:, 5:7]), axis = 1 )   
    trainY = train[:, 3 : 5]

    rf.fit(trainX, trainY)
   

    testX = test[:, 0:3]
    testX = np.concatenate( (testX, test[:, 5:7]), axis = 1 )  
    testY = test[:, 3 : 5]

    y_predict = rf.predict(testX)
    return y_predict, testY

def plot_median_linear_time_step( timeStep, round, location, board ):
  
    if not os.path.exists("my_plots/5_second/Linear" ):
        os.mkdir( "my_plots/5_second/Linear" )
   
    if not os.path.exists("my_plots/5_second/Linear/predicted_o3_Vs_epa_o3" ):
        os.mkdir( "my_plots/5_second/Linear/predicted_o3_Vs_epa_o3" )
        os.mkdir( "my_plots/5_second/Linear/predicted_o3_Vs_epa_o3/round1" )
        os.mkdir( "my_plots/5_second/Linear/predicted_o3_Vs_epa_o3/round2" )
        os.mkdir( "my_plots/5_second/Linear/predicted_o3_Vs_epa_o3/round3" )

    if not os.path.exists("my_plots/5_second/Linear/predicted_no2_Vs_epa_no2" ):
        os.mkdir( "my_plots/5_second/Linear/predicted_no2_Vs_epa_no2" )
        os.mkdir( "my_plots/5_second/Linear/predicted_no2_Vs_epa_no2/round1" )
        os.mkdir( "my_plots/5_second/Linear/predicted_no2_Vs_epa_no2/round2" )
        os.mkdir( "my_plots/5_second/Linear/predicted_no2_Vs_epa_no2/round3" )

    test = second_linear_time_step( timeStep, round, location, board )
    print("plotting round " + str(round) + " " + location + " Board # " + str(board) + " time step of " + str(timeStep) + " ...")
    plt.scatter((test[PREDICT])[:, NO2], (test[ACTUAL])[:, NO2], s=.7)
    plt.plot( range(0, 100), range(0, 100) )
    plt.xlim( 0, 90 )
    plt.ylim( 0, 90 )
    plt.xlabel('predicted no2')
    plt.ylabel('actual no2') 
    plt.title( "actual no2 Vs. predicted no2: " + str(location) + " board " + str(board) )  
    plt.savefig("my_plots/5_second/Linear/" + "predicted_no2_Vs_epa_no2" + "/round" + str(round) + "/" + str(round) + "-" + location + "-" + str(board) + "-" + str(timeStep) + ".png") 
    plt.clf()
    plt.scatter((test[PREDICT])[:, O3], (test[ACTUAL])[:, O3], s=.7)
    plt.plot( range(0, 100), range(0, 100) )
    plt.xlim( 0, 90 )
    plt.ylim( 0, 90 )
 
    plt.xlabel('predicted o3')
    plt.ylabel('actual o3')
    plt.title( "actual o3 Vs. predicted o3: " + str(location) + " board " + str(board))
    plt.savefig("my_plots/5_second/Linear/" + "predicted_o3_Vs_epa_o3" + "/round" + str(round) + "/" + str(round) + "-" + location + "-" + str(board) + "-" + str(timeStep) + ".png") 
    plt.clf()



def mean_squared_error_2Steps( round, location, board ):
    if not os.path.exists("my_plots/5_second/Random_Forest" ):
        os.mkdir( "my_plots/5_second/Random_Forest" )
   
    if not os.path.exists("my_plots/5_second/Random_Forest/o3_mean_squared_error" ):
        os.mkdir( "my_plots/5_second/Random_Forest/o3_mean_squared_error" )
        os.mkdir( "my_plots/5_second/Random_Forest/o3_mean_squared_error/round1" )
        os.mkdir( "my_plots/5_second/Random_Forest/o3_mean_squared_error/round2" )
        os.mkdir( "my_plots/5_second/Random_Forest/o3_mean_squared_error/round3" )

    if not os.path.exists("my_plots/5_second/Random_Forest/no2_mean_squared_error" ):
        os.mkdir( "my_plots/5_second/Random_Forest/no2_mean_squared_error" )
        os.mkdir( "my_plots/5_second/Random_Forest/no2_mean_squared_error/round1" )
        os.mkdir( "my_plots/5_second/Random_Forest/no2_mean_squared_error/round2" )
        os.mkdir( "my_plots/5_second/Random_Forest/no2_mean_squared_error/round3" )
   
    print("round " + str(round) + " " + location + " Board # " + str(board) +  " ...")

 
    data = train3Point( round, location, board)
    mseno2 = mean_squared_error( (data[PREDICT])[:, NO2], (data[ACTUAL])[:, NO2] )
    mseo3 = mean_squared_error( (data[PREDICT])[:, O3], (data[ACTUAL])[:, O3] ) 
 
    return mseno2, mseo3
    
 

no2s = []
o3s = []
f1=open('3Point.txt', 'w+')
data = mean_squared_error_2Steps( 1, 'donovan', 17 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'donovan round 1 board 17', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 1, 'donovan', 21 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'donovan round 1 board 21', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 1, 'elcajon', 11 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'el cajon round 1 board 11', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 1, 'elcajon', 12 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'el cajon round 1 board 12', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 1, 'elcajon', 13 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'el cajon round 1 board 13', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 1, 'shafter', 18 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'shafter round 1 board 18', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 1, 'shafter', 20 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'shafter round 1 board 20', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 2, 'donovan', 18 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'donovan round 2 board 18', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 2, 'donovan', 20 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'donovan round 2 board 20', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 2, 'elcajon', 17 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'el cajon round 2 board 17', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 2, 'elcajon', 21 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'el cajon round 2 board 21', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 2, 'shafter', 11 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'shafter round 2 board 11', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 2, 'shafter', 12 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'shafter round 2 board 12', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 2, 'shafter', 13 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'shafter round 2 board 13', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 3, 'donovan', 11 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'donovan round 3 board 11', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 3, 'donovan', 12 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'donovan round 3 board 12', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 3, 'donovan', 13 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'donovan round 3 board 13', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 3, 'elcajon', 18 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'el cajon round 3 board 18', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 3, 'elcajon', 20 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'el cajon round 3 board 20', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 3, 'shafter', 17 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'shafter round 3 board 17', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)
data = mean_squared_error_2Steps( 3, 'shafter', 21 )
no2s.append( data[0] )
o3s.append( data[1] )
print( 'shafter round 3 board 21', file=f1)
print('mse no2 = ' + str(data[0]), file=f1)
print('mse o3 = ' + str(data[1]), file=f1)

print( 'o3 as follows', file=f1 )
print( 'mean = ' + str(np.mean(o3s)), file=f1 )
print( 'median = ' + str(np.median(o3s)) , file=f1)
print( 'no2 as follows', file=f1 )
print( 'mean = ' + str(np.mean(no2s)), file=f1 )
print( 'median = ' + str(np.median(no2s)), file=f1 )


