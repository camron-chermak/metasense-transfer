import metasense
import data
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

NO2 = 0
O3 = 1
PREDICT = 0
ACTUAL = 1

def loadSecondMatrix(timeStep, round, location, board_id, seed=0):   
     
    TIME_STEP = timeStep
    path = "match_5Second/" + location + "/Round" + str(round) + "/match_round_" + str(round) + "_" + location + "_" + str(board_id) + ".csv"
    data = pd.read_csv(path, parse_dates=True)   

    train, test = train_test_split(data, test_size=0.2, random_state=seed)
    
    train = train.sort_values(by=['Time'])
    mat = train['Time'].values
    mat = np.expand_dims(mat, axis=1)
    mat = np.concatenate( (mat, np.expand_dims(train['no2'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(train['o3'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(train['co'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(train['epa-no2'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(train['epa-o3'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(train['temperature'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(train['absolute-humidity'].values, axis=1)), axis=1)
    for time in range(TIME_STEP):
        timeStep = time + 1 
  #      tempDates = np.chararray((len(train['Time']), 1), itemsize=30)
  #      for index in range(len(tempDates)):
  #          if (index - timeStep) >= 0:
  #              tempDates[index] = train['Time'].values[index - timeStep]
  #          else:
  #              tempDates[index] = "no data"
  #      mat = np.concatenate( (mat, tempDates), axis=1)


        tempDates = train['Time'].values
        tempDates = np.expand_dims(tempDates, axis=1)
        mat = np.concatenate( (mat, tempDates), axis=1)


        temp1 = train['no2'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1) 
      
        temp1 = train['o3'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)

        temp1 = train['co'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)

        temp1 = train['epa-no2'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)

        temp1 = train['epa-o3'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)
 
        temp1 = train['temperature'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)
 
        temp1 = train['absolute-humidity'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)
       
    train = mat
    train = train[TIME_STEP: ]
   
    test = test.sort_values(by=['Time'])
    mat = test['Time'].values
    mat = np.expand_dims(mat, axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['no2'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['o3'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['co'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['epa-no2'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['epa-o3'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['temperature'].values, axis=1)), axis=1)
    mat = np.concatenate( (mat, np.expand_dims(test['absolute-humidity'].values, axis=1)), axis=1)
    for time in range(TIME_STEP):
        timeStep = time + 1
       # tempDates = np.chararray((len(test['Time']), 1), itemsize=30)
       # for index in range(len(tempDates)):
       #     if (index - timeStep) >= 0:
       #         tempDates[index] = test['Time'].values[index - timeStep]
       #     else:
       #         tempDates[index] = "no data"
       # mat = np.concatenate( (mat, tempDates), axis=1)
        tempDates = test['Time'].values
        tempDates = np.expand_dims(tempDates, axis=1)
        mat = np.concatenate( (mat, tempDates), axis=1)
 
        temp1 = test['no2'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)

        temp1 = test['o3'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)

        temp1 = test['co'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)

        temp1 = test['epa-no2'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)
        
        temp1 = test['epa-o3'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)
 
        temp1 = test['temperature'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)

        temp1 = test['absolute-humidity'].values 
        temp1 = np.pad( temp1, (timeStep, 0), 'constant' ) 
        temp1 = temp1[ : (-1 * timeStep) ]  
        temp1 = np.expand_dims(temp1, axis=1)
        mat = np.concatenate( (mat, temp1), axis=1)

       

    test = mat
    test = test[TIME_STEP:]
    
    return train, test


def second_linear_time_step( timeStep, round, location, board ):
    TIME_STEP = timeStep
    lr = linear_model.LinearRegression()
    data = loadSecondMatrix( TIME_STEP, round, location, board)
    train = data[0]
    test = data[1]

    trainX = train[:, 1:4]
    trainX = np.concatenate( (trainX, train[:, 6:8]), axis = 1 )  
    for index in range(TIME_STEP): 
        trainX = np.concatenate( (trainX, train[: , (index + 1)*8 + 1 : (index + 1)*8 + 4]), axis=1)
        trainX = np.concatenate( (trainX, train[: , (index + 1)*8 + 6 : (index + 1)*8 + 8]), axis=1)
    trainY = train[:, 4 : 6]

    lr.fit(trainX, trainY)
   

    testX = test[:, 1:4]
    testX = np.concatenate( (testX, test[:, 6:8]), axis = 1 )  
    for index in range(TIME_STEP): 
        testX = np.concatenate( (testX, test[: , (index + 1)*8 + 1 : (index + 1)*8 + 4]), axis=1)
        testX = np.concatenate( (testX, test[: , (index + 1)*8 + 6 : (index + 1)*8 + 8]), axis=1)
    testY = test[:, 4 : 6]

    y_predict = lr.predict(testX)
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



def plot_mean_squared_error( startTimeStep, endTimeStep, round, location, board ):
    if not os.path.exists("my_plots/5_second/Linear" ):
        os.mkdir( "my_plots/5_second/Linear" )
   
    if not os.path.exists("my_plots/5_second/Linear/o3_mean_squared_error" ):
        os.mkdir( "my_plots/5_second/Linear/o3_mean_squared_error" )
        os.mkdir( "my_plots/5_second/Linear/o3_mean_squared_error/round1" )
        os.mkdir( "my_plots/5_second/Linear/o3_mean_squared_error/round2" )
        os.mkdir( "my_plots/5_second/Linear/o3_mean_squared_error/round3" )

    if not os.path.exists("my_plots/5_second/Linear/no2_mean_squared_error" ):
        os.mkdir( "my_plots/5_second/Linear/no2_mean_squared_error" )
        os.mkdir( "my_plots/5_second/Linear/no2_mean_squared_error/round1" )
        os.mkdir( "my_plots/5_second/Linear/no2_mean_squared_error/round2" )
        os.mkdir( "my_plots/5_second/Linear/no2_mean_squared_error/round3" )
    
    mino3 = 100
    mino3Index = startTimeStep
    minno2 = 100
    minno2Index = startTimeStep
    no2timeStepErrors = []
    o3timeStepErrors = []
    for index in range( startTimeStep, endTimeStep + 1 ):
        print(index)
        data = second_linear_time_step(index, round, location, board)
        mseno2 = mean_squared_error( (data[PREDICT])[:, NO2], (data[ACTUAL])[:, NO2] )
        mseo3 = mean_squared_error( (data[PREDICT])[:, O3], (data[ACTUAL])[:, O3] )
        #if index == startTimeStep:
        #    minno2 = mseno2
        #    mino3 = mseo3
        #elif mseno2 < minno2:
        #    minno2 = mseno2
        #    minno2Index = index
        #elif mseo3 < mino3:
        #    mino3 = mseo3
        #    mino3Index = index
        no2timeStepErrors.append( mseno2 )
        o3timeStepErrors.append( mseo3 )

    minno2 = min(no2timeStepErrors)
    minno2Index = no2timeStepErrors.index(minno2)
    mino3 = min(o3timeStepErrors)
    mino3Index = o3timeStepErrors.index(mino3)

    print( "min no2 error: " + str(minno2) + " at time step of " + str(minno2Index)) 
    print( "min o3 error: " + str(mino3) + " at time step of " + str(mino3Index)) 
    plt.plot( range(startTimeStep, endTimeStep + 1), no2timeStepErrors)
    #plt.plot( range(0, 100), range(0, 100) )
    #plt.xlim( 0, 90 )
    #plt.ylim( 0, 90 )
    plt.xlabel('time step')
    plt.ylabel('no2 mean squared error') 
    plt.title( "no2 mean squared error: " + str(location) + " board " + str(board) + " steps: " + str(startTimeStep) + " to " + str(endTimeStep) )  
    plt.savefig("my_plots/5_second/Linear/" + "no2_mean_squared_error" + "/round" + str(round) + "/" + str(round) + "-" + location + "-" + str(board) + "_" + str(startTimeStep) + "-" + str(endTimeStep) + ".png") 
    plt.clf() 

    plt.plot( range(startTimeStep, endTimeStep + 1), o3timeStepErrors)
    #plt.plot( range(0, 100), range(0, 100) )
    #plt.xlim( 0, 90 )
    #plt.ylim( 0, 90 )
    plt.xlabel('time step')
    plt.ylabel('o3 mean squared error') 
    plt.title( "o3 mean squared error: " + str(location) + " board " + str(board) + " steps: " + str(startTimeStep) + " to " + str(endTimeStep) )  
    plt.savefig("my_plots/5_second/Linear/" + "o3_mean_squared_error" + "/round" + str(round) + "/" + str(round) + "-" + location + "-" + str(board) + "_" + str(startTimeStep) + "-" + str(endTimeStep) + ".png") 
    plt.clf() 
    
 

plot_mean_squared_error(0, 10, 1, 'donovan', 17)
#plot_mean_squared_error(0, 100, 1, 'elcajon', 11)

#for time in range(4):
   # plot_median_linear_time_step( time, 1, 'donovan', 17 ) 
   # plot_median_linear_time_step( time, 1, 'donovan', 19 )
   # plot_median_linear_time_step( time, 1, 'donovan', 21 )
   # plot_median_linear_time_step( time, 1, 'elcajon', 11 )
   # plot_median_linear_time_step( time, 1, 'elcajon', 12 )
   # plot_median_linear_time_step( time, 1, 'elcajon', 13 )
   # plot_median_linear_time_step( time, 1, 'shafter', 15 )
   # plot_median_linear_time_step( time, 1, 'shafter', 18 )
   # plot_median_linear_time_step( time, 1, 'shafter', 20 )
   # plot_median_linear_time_step( time, 2, 'donovan', 15 )
   # plot_median_linear_time_step( time, 2, 'donovan', 18 )
   # plot_median_linear_time_step( time, 2, 'donovan', 20 )
   # plot_median_linear_time_step( time, 2, 'elcajon', 17 )
   # plot_median_linear_time_step( time, 2, 'elcajon', 19 )
   # plot_median_linear_time_step( time, 2, 'elcajon', 21 )
   # plot_median_linear_time_step( time, 2, 'shafter', 11 )
   # plot_median_linear_time_step( time, 2, 'shafter', 12 )
   # plot_median_linear_time_step( time, 2, 'shafter', 13 )
   # plot_median_linear_time_step( time, 3, 'donovan', 11 )
   # plot_median_linear_time_step( time, 3, 'donovan', 12 )
   # plot_median_linear_time_step( time, 3, 'donovan', 13 )
   # plot_median_linear_time_step( time, 3, 'elcajon', 15 )
   # plot_median_linear_time_step( time, 3, 'elcajon', 18 )
   # plot_median_linear_time_step( time, 3, 'elcajon', 20 )
   # plot_median_linear_time_step( time, 3, 'shafter', 17 ) 
   # plot_median_linear_time_step( time, 3, 'shafter', 19 ) 
   # plot_median_linear_time_step( time, 3, 'shafter', 21 ) 




