import metasense
import data
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sklearn
#from sklearn import linear_model
#from sklearn.linear_model import LinearRegression
from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_predict
import pandas as pd
from sklearn.model_selection import train_test_split

NO2 = 0
O3 = 1
PREDICT = 0
ACTUAL = 1

def loadMedianMatrix(timeStep, round, location, board_id, seed=0):   
     
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
    for time in range(timeStep):
        timeStep = time + 1
        tempDates = np.chararray((len(train['Time']), 1), itemsize=30)
        for index in range(len(tempDates)):
            if (index - timeStep) >= 0:
                tempDates[index] = train['Time'].values[index - timeStep]
            else:
                tempDates[index] = "no data"
        mat = np.concatenate( (mat, tempDates), axis=1)
        temp = np.zeros(shape=(len(train['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['no2'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['o3'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['co'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['epa-no2'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['epa-o3'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['temperature'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = train['absolute-humidity'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(train['Time']), 1))
    train = mat
   
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
    for time in range(timeStep):
        timeStep = time + 1
        tempDates = np.chararray((len(test['Time']), 1), itemsize=30)
        for index in range(len(tempDates)):
            if (index - timeStep) >= 0:
                tempDates[index] = test['Time'].values[index - timeStep]
            else:
                tempDates[index] = "no data"
        mat = np.concatenate( (mat, tempDates), axis=1)
        temp = np.zeros(shape=(len(test['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['no2'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['o3'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['co'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['epa-no2'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['epa-o3'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['Ts'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['Time']), 1))
        for index in range(len(temp)):
            if (index - timeStep) >= 0:
                temp[index] = test['absolute-humidity'].values[index - timeStep]
        mat = np.concatenate( (mat, temp), axis=1)
        temp = np.zeros(shape=(len(test['Time']), 1))

    test = mat
    
    return train, test


def median_kernel_time_step( timeStep, round, location, board ):
    TIME_STEP = timeStep
    kr = kernel_ridge.KernelRidge()
    print('create')
    data = loadMedianMatrix( TIME_STEP, round, location, board)
    train = data[0]
    test = data[1]
    trainX = train[TIME_STEP::, 1]
    trainX = np.expand_dims(trainX, axis=1)
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, 2], axis=1)), axis=1)
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, 3], axis=1)), axis=1)
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, 6], axis=1)), axis=1)
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, 7], axis=1)), axis=1)
    for index in range(TIME_STEP):
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, 1 + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, 2 + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, 3 + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, 6 + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, 7 + (index + 1)*8], axis=1)), axis=1)
    trainY = train[TIME_STEP::,4]
    trainY = np.expand_dims(trainY, axis=1)
    trainY = np.concatenate( (trainY, np.expand_dims(train[TIME_STEP::, 5], axis=1)), axis=1)
    kr.fit(trainX, trainY)
    print('fit') 
   
    testX = test[TIME_STEP::, 1]
    testX = np.expand_dims(testX, axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, 2], axis=1)), axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, 3], axis=1)), axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, 6], axis=1)), axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, 7], axis=1)), axis=1)
    for index in range(TIME_STEP):
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, 1 + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, 2 + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, 3 + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, 6 + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, 7 + (index + 1)*8], axis=1)), axis=1)
    testY = test[TIME_STEP::,4]
    testY = np.expand_dims(testY, axis=1)
    testY = np.concatenate( (testY, np.expand_dims(test[TIME_STEP::, 5], axis=1)), axis=1) 
    y_predict = kr.predict(testX)
    return y_predict, testY

def plot_median_kernel_time_step( timeStep, round, location, board ):
  
    if not os.path.exists("my_plots/Median/Kernel" ):
        os.mkdir( "my_plots/Median/Kernel" )
   
    if not os.path.exists("my_plots/Median/Kernel/predicted_o3_Vs_epa_o3" ):
        os.mkdir( "my_plots/Median/Kernel/predicted_o3_Vs_epa_o3" )
        os.mkdir( "my_plots/Median/Kernel/predicted_o3_Vs_epa_o3/round1" )
        os.mkdir( "my_plots/Median/Kernel/predicted_o3_Vs_epa_o3/round2" )
        os.mkdir( "my_plots/Median/Kernel/predicted_o3_Vs_epa_o3/round3" )

    if not os.path.exists("my_plots/Median/Kernel/predicted_no2_Vs_epa_no2" ):
        os.mkdir( "my_plots/Median/Kernel/predicted_no2_Vs_epa_no2" )
        os.mkdir( "my_plots/Median/Kernel/predicted_no2_Vs_epa_no2/round1" )
        os.mkdir( "my_plots/Median/Kernel/predicted_no2_Vs_epa_no2/round2" )
        os.mkdir( "my_plots/Median/Kernel/predicted_no2_Vs_epa_no2/round3" )

    test = median_kernel_time_step( timeStep, round, location, board )
    print("plotting round " + str(round) + " " + location + " Board # " + str(board) + " ...")
    plt.scatter((test[PREDICT])[:, NO2], (test[ACTUAL])[:, NO2], s=.7)
 
    #minVal = np.min( np.append( np.min((test[PREDICT])[:, NO2]), np.min((test[ACTUAL])[:, NO2])))
    #maxVal = np.max( np.append( np.max((test[PREDICT])[:, NO2]), np.max((test[ACTUAL])[:, NO2]))) 
    #plt.plot( range(int(minVal) - 1, int(maxVal) + 1), range(int(minVal) - 1, int(maxVal) + 1) )
    plt.plot( range(0, 100), range(0, 100) )
    plt.xlim( 0, 90 )
    plt.ylim( 0, 90 )
    plt.xlabel('predicted no2')
    plt.ylabel('actual no2') 
    plt.title( "actual no2 Vs. predicted no2: " + str(location) + " board " + str(board) )  
    plt.savefig("my_plots/Median/Kernel/" + "predicted_no2_Vs_epa_no2" + "/round" + str(round) + "/" + str(round) + "-" + location + "-" + str(board) + "-" + str(timeStep) + ".png") 
    plt.clf()
    plt.scatter((test[PREDICT])[:, O3], (test[ACTUAL])[:, O3], s=.7)
    #minVal = np.min( np.append( np.min((test[PREDICT])[:, O3]), np.min((test[ACTUAL])[:, O3])))
    #maxVal = np.max( np.append( np.max((test[PREDICT])[:, O3]), np.max((test[ACTUAL])[:, O3]))) 
    #plt.plot( range(int(minVal) - 1, int(maxVal) + 1), range(int(minVal) - 1, int(maxVal) + 1) )
    plt.plot( range(0, 100), range(0, 100) )
    plt.xlim( 0, 90 )
    plt.ylim( 0, 90 )
 
    plt.xlabel('predicted o3')
    plt.ylabel('actual o3')
    plt.title( "actual o3 Vs. predicted o3: " + str(location) + " board " + str(board))
    plt.savefig("my_plots/Median/Kernel/" + "predicted_o3_Vs_epa_o3" + "/round" + str(round) + "/" + str(round) + "-" + location + "-" + str(board) + "-" + str(timeStep) + ".png") 
    plt.clf()


time = 0 
plot_median_kernel_time_step( time, 1, 'donovan', 17 ) #DONE
#plot_median_linear_time_step( time, 1, 'donovan', 19 ) #DONE
#plot_median_linear_time_step( time, 1, 'donovan', 21 ) #DONE
#plot_median_linear_time_step( time, 1, 'elcajon', 11 )
#plot_median_linear_time_step( time, 1, 'elcajon', 12 )
#plot_median_linear_time_step( time, 1, 'elcajon', 13 )
#plot_median_linear_time_step( time, 1, 'shafter', 15 )
#plot_median_linear_time_step( time, 1, 'shafter', 18 )
#plot_median_linear_time_step( time, 1, 'shafter', 20 )
#plot_median_linear_time_step( time, 2, 'donovan', 15 )
#plot_median_linear_time_step( time, 2, 'donovan', 18 )
#plot_median_linear_time_step( time, 2, 'donovan', 20 )
#plot_median_linear_time_step( time, 2, 'elcajon', 17 )
#plot_median_linear_time_step( time, 2, 'elcajon', 19 )
#plot_median_linear_time_step( time, 2, 'elcajon', 21 )
#plot_median_linear_time_step( time, 2, 'shafter', 11 )
#plot_median_linear_time_step( time, 2, 'shafter', 12 )
#plot_median_linear_time_step( time, 2, 'shafter', 13 )
#plot_median_linear_time_step( time, 3, 'donovan', 11 )
#plot_median_linear_time_step( time, 3, 'donovan', 12 )
#plot_median_linear_time_step( time, 3, 'donovan', 13 )
#plot_median_linear_time_step( time, 3, 'elcajon', 15 )
#plot_median_linear_time_step( time, 3, 'elcajon', 18 )
#plot_median_linear_time_step( time, 3, 'elcajon', 20 )
#plot_median_linear_time_step( time, 3, 'shafter', 17 ) #TODO
#plot_median_linear_time_step( time, 3, 'shafter', 19 ) #TODO
#plot_median_linear_time_step( time, 3, 'shafter', 21 ) #TODO




