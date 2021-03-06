import metasense
import metasense.data
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_predict
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

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


def second_linear_time_step( timeStep, round1, location1, round2, location2, board ):
    TIME_STEP = timeStep
    rf = RandomForestRegressor(random_state=0)
    data = loadSecondMatrix( TIME_STEP, round1, location1, board)
    cross = loadSecondMatrix( TIME_STEP, round2, location2, board)
    train = data[0]
    train = np.concatenate((train, data[1]), axis=0)
    test = cross[0]
    test = np.concatenate((train, cross[1]), axis=0)
    trainX = train[:, 1:4]
    trainX = np.concatenate( (trainX, train[:, 6:8]), axis = 1 )  
    for index in range(TIME_STEP): 
        trainX = np.concatenate( (trainX, train[: , (index + 1)*8 + 1 : (index + 1)*8 + 4]), axis=1)
        trainX = np.concatenate( (trainX, train[: , (index + 1)*8 + 6 : (index + 1)*8 + 8]), axis=1)
    trainY = train[:, 4 : 6]

    rf.fit(trainX, trainY)
   

    testX = test[:, 1:4]
    testX = np.concatenate( (testX, test[:, 6:8]), axis = 1 )  
    for index in range(TIME_STEP): 
        testX = np.concatenate( (testX, test[: , (index + 1)*8 + 1 : (index + 1)*8 + 4]), axis=1)
        testX = np.concatenate( (testX, test[: , (index + 1)*8 + 6 : (index + 1)*8 + 8]), axis=1)
    testY = test[:, 4 : 6]

    y_predict = rf.predict(testX)
    return y_predict, testY

def plot_random_forest_cross_loc_time_step( timeStep, round1, location1, round2, location2, board ):
  
    if not os.path.exists("my_plots/5_second/Random_Forest" ):
        os.mkdir( "my_plots/5_second/Random_Forest" )
   
    if not os.path.exists("my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3" ):
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/round1" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/round2" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/round3" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Location" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Location/Train_donovan" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Location/Train_elcajon" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Location/Train_shafter" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Location_2" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Location_2/Test_donovan" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Location_2/Test_elcajon" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Location_2/Test_shafter" )

    if not os.path.exists("my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2" ):
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/round1" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/round2" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/round3" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Location" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Location/Train_donovan" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Location/Train_elcajon" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Location/Train_shafter" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Location_2" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Location_2/Test_donovan" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Location_2/Test_elcajon" )
        os.mkdir( "my_plots/5_second/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Location_2/Test_shafter" )

    test = second_linear_time_step( timeStep, round1, location1, round2, location2, board )
    print("plotting " + location1 + " to " + location2 + "Board # " + str(board) + " time step of " + str(timeStep) + " ...")
    plt.scatter((test[PREDICT])[:, NO2], (test[ACTUAL])[:, NO2], s=.7)
    plt.plot( range(0, 100), range(0, 100) )
    plt.xlim( 0, 90 )
    plt.ylim( 0, 90 )
    plt.xlabel('predicted no2')
    plt.ylabel('actual no2') 
    plt.title( "no2: " + location1 + " to " + location2 + " Board # " + str(board) )  
    plt.savefig("my_plots/5_second/Random_Forest/" + "predicted_no2_Vs_epa_no2" + "/Cross_Location/Train_" + location1 + "/" + location1 + "_" + location2 + "_" + str(board) + "-" + str(timeStep) + ".png") 
    plt.clf()
    plt.scatter((test[PREDICT])[:, O3], (test[ACTUAL])[:, O3], s=.7)
    plt.plot( range(0, 100), range(0, 100) )
    plt.xlim( 0, 90 )
    plt.ylim( 0, 90 )
 
    plt.xlabel('predicted o3')
    plt.ylabel('actual o3')
    plt.title( "o3: " + location1 + " to " + location2 + " Board # " + str(board) )  
    plt.savefig("my_plots/5_second/Random_Forest/" + "predicted_o3_Vs_epa_o3" + "/Cross_Location/Train_" + location1 + "/" + location1 + "_" + location2 + "_" + str(board) + "-" + str(timeStep) + ".png")  
    plt.clf()



def plot_mean_squared_error( startTimeStep, endTimeStep, round1, location1, round2, location2, board ):
    if not os.path.exists("my_plots/5_second/Random_Forest" ):
        os.mkdir( "my_plots/5_second/Random_Forest" )
    
    if not os.path.exists("my_plots/5_second/Random_Forest/o3_mean_squared_error" ):
        os.mkdir( "my_plots/5_second/Random_Forest/o3_mean_squared_error" )
   
    if not os.path.exists("my_plots/5_second/Random_Forest/no2_mean_squared_error" ):
        os.mkdir( "my_plots/5_second/Random_Forest/no2_mean_squared_error" )

    if not os.path.exists("my_plots/5_second/Random_Forest/o3_mean_squared_error/Level1" ):
        os.mkdir( "my_plots/5_second/Random_Forest/o3_mean_squared_error/Level1" )
        os.mkdir( "my_plots/5_second/Random_Forest/o3_mean_squared_error/Level1/Test_donovan" )
        os.mkdir( "my_plots/5_second/Random_Forest/o3_mean_squared_error/Level1/Test_elcajon" )
        os.mkdir( "my_plots/5_second/Random_Forest/o3_mean_squared_error/Level1/Test_shafter" )

    if not os.path.exists("my_plots/5_second/Random_Forest/no2_mean_squared_error/Level1" ):
        os.mkdir( "my_plots/5_second/Random_Forest/no2_mean_squared_error/Level1" )
        os.mkdir( "my_plots/5_second/Random_Forest/no2_mean_squared_error/Level1/Test_donovan" )
        os.mkdir( "my_plots/5_second/Random_Forest/no2_mean_squared_error/Level1/Test_elcajon" )
        os.mkdir( "my_plots/5_second/Random_Forest/no2_mean_squared_error/Level1/Test_shafter" )
   
    print("plotting " + location1 + " to " + location2 + " Board # " + str(board) +  " ...")

 
    mino3 = 100
    mino3Index = startTimeStep
    minno2 = 100
    minno2Index = startTimeStep
    no2timeStepErrors = []
    o3timeStepErrors = []
    for index in range( startTimeStep, endTimeStep + 1 ):
        print(index)
        data = second_linear_time_step(index, round1, location1, round2, location2, board)
        mseno2 = mean_squared_error( (data[PREDICT])[:, NO2], (data[ACTUAL])[:, NO2] )
        mseo3 = mean_squared_error( (data[PREDICT])[:, O3], (data[ACTUAL])[:, O3] )
        no2timeStepErrors.append( mseno2 )
        o3timeStepErrors.append( mseo3 )

    mino3 = min(o3timeStepErrors)
    minno2 = min(no2timeStepErrors)
    mino3Index = o3timeStepErrors.index(mino3)
    minno2Index = no2timeStepErrors.index(minno2)
    print( "min no2 error: " + str(minno2) + " at time step of " + str(minno2Index + startTimeStep)) 
    print( "min o3 error: " + str(mino3) + " at time step of " + str(mino3Index + startTimeStep)) 
    plt.plot( range(startTimeStep, endTimeStep + 1), no2timeStepErrors) 
    plt.xlabel('time step')
    plt.ylabel('no2 mean squared error') 
    plt.title( "no2 mean squared error: " + location1 + " to " + location2 + " board " + str(board) )  
    plt.savefig("my_plots/5_second/Random_Forest/no2_mean_squared_error/Level1/Test_" + location2 + "/" + location1 + "-" + location2 + "_" +  str(board) + "_" + str(startTimeStep) + "-" + str(endTimeStep) + ".png") 
    plt.clf() 

    plt.plot( range(startTimeStep, endTimeStep + 1), o3timeStepErrors) 
    plt.xlabel('time step')
    plt.ylabel('o3 mean squared error') 
    plt.title( "o3 mean squared error: " + location1 + " to " + location2 + " board " + str(board) )  
    plt.savefig("my_plots/5_second/Random_Forest/o3_mean_squared_error/Level1/Test_" + location2 + "/" + location1 + "-" + location2 + "_" +  str(board) + "_" + str(startTimeStep) + "-" + str(endTimeStep) + ".png")  
    plt.clf() 

    return minno2Index, mino3Index, minno2, mino3
    
 


def random_forest_cross_loc_plotter_5Second( time ):
    print("Boards 17, 19, 21")
    plot_random_forest_cross_loc_time_step( time, 1, 'donovan', 2, 'elcajon', 17 )
    plot_random_forest_cross_loc_time_step( time, 1, 'donovan', 3, 'shafter', 17 )
    plot_random_forest_cross_loc_time_step( time, 1, 'donovan', 2, 'elcajon', 19 )
    plot_random_forest_cross_loc_time_step( time, 1, 'donovan', 3, 'shafter', 19 )
    plot_random_forest_cross_loc_time_step( time, 1, 'donovan', 2, 'elcajon', 21 )
    plot_random_forest_cross_loc_time_step( time, 1, 'donovan', 3, 'shafter', 21 )

    plot_random_forest_cross_loc_time_step( time, 2, 'elcajon', 1, 'donovan', 17 )
    plot_random_forest_cross_loc_time_step( time, 2, 'elcajon', 3, 'shafter', 17 )
    plot_random_forest_cross_loc_time_step( time, 2, 'elcajon', 1, 'donovan', 19 )
    plot_random_forest_cross_loc_time_step( time, 2, 'elcajon', 3, 'shafter', 19 )
    plot_random_forest_cross_loc_time_step( time, 2, 'elcajon', 1, 'donovan', 21 )
    plot_random_forest_cross_loc_time_step( time, 2, 'elcajon', 3, 'shafter', 21 )

    plot_random_forest_cross_loc_time_step( time, 3, 'shafter', 1, 'donovan', 17 )
    plot_random_forest_cross_loc_time_step( time, 3, 'shafter', 2, 'elcajon', 17 )
    plot_random_forest_cross_loc_time_step( time, 3, 'shafter', 1, 'donovan', 19 )
    plot_random_forest_cross_loc_time_step( time, 3, 'shafter', 2, 'elcajon', 19 )
    plot_random_forest_cross_loc_time_step( time, 3, 'shafter', 1, 'donovan', 21 )
    plot_random_forest_cross_loc_time_step( time, 3, 'shafter', 2, 'elcajon', 21 )


    print("Boards 11, 12, 13")
    plot_random_forest_cross_loc_time_step( time, 1, 'elcajon', 2, 'shafter', 11 )
    plot_random_forest_cross_loc_time_step( time, 1, 'elcajon', 3, 'donovan', 11 )
    plot_random_forest_cross_loc_time_step( time, 1, 'elcajon', 2, 'shafter', 12 )
    plot_random_forest_cross_loc_time_step( time, 1, 'elcajon', 3, 'donovan', 12 )
    plot_random_forest_cross_loc_time_step( time, 1, 'elcajon', 2, 'shafter', 13 )
    plot_random_forest_cross_loc_time_step( time, 1, 'elcajon', 3, 'donovan', 13 )

    plot_random_forest_cross_loc_time_step( time, 2, 'shafter', 1, 'elcajon', 11 )
    plot_random_forest_cross_loc_time_step( time, 2, 'shafter', 3, 'donovan', 11 )
    plot_random_forest_cross_loc_time_step( time, 2, 'shafter', 1, 'elcajon', 12 )
    plot_random_forest_cross_loc_time_step( time, 2, 'shafter', 3, 'donovan', 12 )
    plot_random_forest_cross_loc_time_step( time, 2, 'shafter', 1, 'elcajon', 13 )
    plot_random_forest_cross_loc_time_step( time, 2, 'shafter', 3, 'donovan', 13 )

    plot_random_forest_cross_loc_time_step( time, 3, 'donovan', 1, 'elcajon', 11 )
    plot_random_forest_cross_loc_time_step( time, 3, 'donovan', 2, 'shafter', 11 )
    plot_random_forest_cross_loc_time_step( time, 3, 'donovan', 1, 'elcajon', 12 )
    plot_random_forest_cross_loc_time_step( time, 3, 'donovan', 2, 'shafter', 12 )
    plot_random_forest_cross_loc_time_step( time, 3, 'donovan', 1, 'elcajon', 13 )
    plot_random_forest_cross_loc_time_step( time, 3, 'donovan', 2, 'shafter', 13 )

    print("Boards 15, 18, 20")
    plot_random_forest_cross_loc_time_step( time, 1, 'shafter', 2, 'donovan', 15 )
    plot_random_forest_cross_loc_time_step( time, 1, 'shafter', 3, 'elcajon', 15 )
    plot_random_forest_cross_loc_time_step( time, 1, 'shafter', 2, 'donovan', 18 )
    plot_random_forest_cross_loc_time_step( time, 1, 'shafter', 3, 'elcajon', 18 )

    plot_random_forest_cross_loc_time_step( time, 2, 'donovan', 1, 'shafter', 15 )
    plot_random_forest_cross_loc_time_step( time, 2, 'donovan', 3, 'elcajon', 15 )
    plot_random_forest_cross_loc_time_step( time, 2, 'donovan', 1, 'shafter', 18 )
    plot_random_forest_cross_loc_time_step( time, 2, 'donovan', 3, 'elcajon', 18 )

    plot_random_forest_cross_loc_time_step( time, 3, 'elcajon', 1, 'shafter', 15 )
    plot_random_forest_cross_loc_time_step( time, 3, 'elcajon', 2, 'donovan', 15 )
    plot_random_forest_cross_loc_time_step( time, 3, 'elcajon', 1, 'shafter', 18 )
    plot_random_forest_cross_loc_time_step( time, 3, 'elcajon', 2, 'donovan', 18 )




def random_forest_cross_loc_error_plotter_5Second( start, end ):
    minno2Steps = []
    mino3Steps = []
    minno2 = []
    mino3 = []

    print("Boards 17, 19, 21")
    minIndex = plot_mean_squared_error( start, end, 1, 'donovan', 2, 'elcajon', 17 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'donovan', 3, 'shafter', 17 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'donovan', 2, 'elcajon', 19 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'donovan', 3, 'shafter', 19 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'donovan', 2, 'elcajon', 21 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'donovan', 3, 'shafter', 21 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 

    minIndex = plot_mean_squared_error( start, end, 2, 'elcajon', 1, 'donovan', 17 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'elcajon', 3, 'shafter', 17 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'elcajon', 1, 'donovan', 19 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'elcajon', 3, 'shafter', 19 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'elcajon', 1, 'donovan', 21 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'elcajon', 3, 'shafter', 21 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
 
    minIndex = plot_mean_squared_error( start, end, 3, 'shafter', 1, 'donovan', 17 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'shafter', 2, 'elcajon', 17 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'shafter', 1, 'donovan', 19 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'shafter', 2, 'elcajon', 19 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'shafter', 1, 'donovan', 21 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'shafter', 2, 'elcajon', 21 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
 
    print("Boards 11, 12, 13")
    minIndex = plot_mean_squared_error( start, end, 1, 'elcajon', 2, 'shafter', 11 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'elcajon', 3, 'donovan', 11 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'elcajon', 2, 'shafter', 12 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'elcajon', 3, 'donovan', 12 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'elcajon', 2, 'shafter', 13 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'elcajon', 3, 'donovan', 13 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 

    minIndex = plot_mean_squared_error( start, end, 2, 'shafter', 1, 'elcajon', 11 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'shafter', 3, 'donovan', 11 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'shafter', 1, 'elcajon', 12 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'shafter', 3, 'donovan', 12 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'shafter', 1, 'elcajon', 12 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'shafter', 3, 'donovan', 12 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 

    minIndex = plot_mean_squared_error( start, end, 3, 'donovan', 1, 'elcajon', 11 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'donovan', 2, 'shafter', 11 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'donovan', 1, 'elcajon', 12 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'donovan', 2, 'shafter', 12 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'donovan', 1, 'elcajon', 13 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'donovan', 2, 'shafter', 13 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 


    print("Boards 15, 18, 20")
    minIndex = plot_mean_squared_error( start, end, 1, 'shafter', 2, 'donovan', 15 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'shafter', 3, 'elcajon', 15 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'shafter', 2, 'donovan', 18 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 1, 'shafter', 3, 'elcajon', 18 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 

    minIndex = plot_mean_squared_error( start, end, 2, 'donovan', 1, 'shafter', 15 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'donovan', 3, 'elcajon', 15 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'donovan', 1, 'shafter', 18 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 2, 'donovan', 3, 'elcajon', 18 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 

    minIndex = plot_mean_squared_error( start, end, 3, 'elcajon', 1, 'shafter', 15 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'elcajon', 2, 'donovan', 15 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'elcajon', 1, 'shafter', 18 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
    minIndex = plot_mean_squared_error( start, end, 3, 'elcajon', 2, 'donovan', 18 )
    minno2Steps.append( minIndex[NO2] )
    mino3Steps.append( minIndex[O3] ) 
    minno2.append( minIndex[2] )
    mino3.append( minIndex[3] ) 
 

    print( "o3 as follows" )
    print(mino3Steps)
    print( "mean = " + str(np.mean(mino3Steps)) )
    print( "median = " + str(np.median(mino3Steps)) )
    print( "mean = " + str(np.mean(mino3)) )
    print( "median = " + str(np.median(mino3)) )
    print( "no2 as follows" )
    print(minno2Steps)
    print( "mean = " + str(np.mean(minno2Steps)) )
    print( "median = " + str(np.median(minno2Steps)) )
    print( "mean = " + str(np.mean(minno2)) )
    print( "median = " + str(np.median(minno2)) )



random_forest_cross_loc_error_plotter_5Second( 0, 25 )
