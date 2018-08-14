import metasense
import metasense.data
from metasense.data import loadMatrix
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict



NO2 = 0
O3 = 1
PREDICT = 0
ACTUAL = 1
NO2_INDEX = 1
O3_INDEX = 2
CO_INDEX = 3
EPA_NO2_INDEX = 4
EPA_O3_INDEX = 5
TEMP_INDEX = 6
HUM_INDEX = 7




def linear_time_step( timeStep, round, location, board ):
    TIME_STEP = timeStep
    lr = linear_model.LinearRegression() 
    data = loadMatrix( TIME_STEP, round, location, board)
    train = data[0]
    test = data[1]
    trainX = train[TIME_STEP::, NO2_INDEX]
    trainX = np.expand_dims(trainX, axis=1) 
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, O3_INDEX], axis=1)), axis=1)
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, TEMP_INDEX], axis=1)), axis=1)
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, HUM_INDEX], axis=1)), axis=1)
    for index in range(TIME_STEP):
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, NO2_INDEX + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, O3_INDEX + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, TEMP_INDEX + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, HUM_INDEX + (index + 1)*8], axis=1)), axis=1)
    trainY = train[TIME_STEP::, CO_INDEX]
    trainY = np.expand_dims(trainY, axis=1)
    lr.fit(trainX, trainY) 
    
    testX = test[TIME_STEP::, NO2_INDEX]
    testX = np.expand_dims(testX, axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, O3_INDEX], axis=1)), axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, TEMP_INDEX], axis=1)), axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, HUM_INDEX], axis=1)), axis=1)
    for index in range(TIME_STEP):
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, NO2_INDEX + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, O3_INDEX + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, TEMP_INDEX + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, HUM_INDEX + (index + 1)*8], axis=1)), axis=1)
    testY = test[TIME_STEP::, CO_INDEX]
    testY = np.expand_dims(testY, axis=1) 
    y_predict = lr.predict(testX)
    return y_predict, testY

def plot_linear_time_step( timeStep, round, location, board ):
    if not os.path.exists("my_plots/Mean"):
        os.mkdir( "my_plots/Mean" )
    
    if not os.path.exists("my_plots/Mean/Linear"):
        os.mkdir( "my_plots/Mean/Linear" )

    if not os.path.exists("my_plots/Mean/Linear/predicting_co" ):
        os.mkdir( "my_plots/Mean/Linear/predicting_co" )
        os.mkdir( "my_plots/Mean/Linear/predicting_co/round1" )
        os.mkdir( "my_plots/Mean/Linear/predicting_co/round2" )
        os.mkdir( "my_plots/Mean/Linear/predicting_co/round3" )

    test = linear_time_step( timeStep, round, location, board )
    print("plotting round " + str(round) + " " + location + " Board # " + str(board))
    plt.scatter((test[PREDICT]), (test[ACTUAL]), s=.7)
    plt.xlabel('predicted co')
    plt.ylabel('actual co')
    myMin = int(min(min(test[PREDICT]), min(test[ACTUAL])))
    myMax = int(max(max(test[PREDICT]), max(test[ACTUAL]))) 
    plt.plot( range(myMin, myMax), range(myMin, myMax) )
    plt.xlim( myMin, myMax )
    plt.ylim( myMin, myMax ) 
    plt.title( "actual co Vs. predicted co: " + str(location) + " board " + str(board) ) 
    plt.savefig("my_plots/Mean/Linear/predicting_co/round" + str(round) + "/" + str(round) + "-" + location + "-" + str(board) + "-" + str(timeStep) + ".png") 
    plt.clf()


time = 0
plot_linear_time_step( time, 1, 'donovan', 17 )
plot_linear_time_step( time, 1, 'donovan', 19 )
plot_linear_time_step( time, 1, 'donovan', 21 )
plot_linear_time_step( time, 1, 'elcajon', 11 )
plot_linear_time_step( time, 1, 'elcajon', 12 )
plot_linear_time_step( time, 1, 'elcajon', 13 )
plot_linear_time_step( time, 1, 'shafter', 15 )
plot_linear_time_step( time, 1, 'shafter', 18 )
plot_linear_time_step( time, 2, 'donovan', 15 )
plot_linear_time_step( time, 2, 'donovan', 18 )
plot_linear_time_step( time, 2, 'donovan', 20 )
plot_linear_time_step( time, 2, 'elcajon', 17 )
plot_linear_time_step( time, 2, 'elcajon', 19 )
plot_linear_time_step( time, 2, 'elcajon', 21 )
plot_linear_time_step( time, 2, 'shafter', 11 )
plot_linear_time_step( time, 2, 'shafter', 12 )
plot_linear_time_step( time, 2, 'shafter', 13 )
plot_linear_time_step( time, 3, 'donovan', 11 )
plot_linear_time_step( time, 3, 'donovan', 12 )
plot_linear_time_step( time, 3, 'donovan', 13 )
plot_linear_time_step( time, 3, 'elcajon', 15 )
plot_linear_time_step( time, 3, 'elcajon', 18 )
plot_linear_time_step( time, 3, 'elcajon', 20 )
plot_linear_time_step( time, 3, 'shafter', 17 )
plot_linear_time_step( time, 3, 'shafter', 19 )
plot_linear_time_step( time, 3, 'shafter', 21 )



