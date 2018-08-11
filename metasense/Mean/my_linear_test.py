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


def linear_time_step( timeStep, round, location, board ):
    TIME_STEP = timeStep
    lr = linear_model.LinearRegression() 
    data = loadMatrix( TIME_STEP, round, location, board)
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
    lr.fit(trainX, trainY) 
    
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
    y_predict = lr.predict(testX)
    return y_predict, testY

def plot_linear_time_step( timeStep, round, location, board ):
    if not os.path.exists("my_plots/Mean"):
        os.mkdir( "my_plots/Mean" )
    
    if not os.path.exists("my_plots/Mean/Linear"):
        os.mkdir( "my_plots/Mean/Linear" )

    if not os.path.exists("my_plots/Mean/Linear/predicted_o3_Vs_epa_o3" ):
        os.mkdir( "my_plots/Mean/Linear/predicted_o3_Vs_epa_o3" )
        os.mkdir( "my_plots/Mean/Linear/predicted_o3_Vs_epa_o3/round1" )
        os.mkdir( "my_plots/Mean/Linear/predicted_o3_Vs_epa_o3/round2" )
        os.mkdir( "my_plots/Mean/Linear/predicted_o3_Vs_epa_o3/round3" )

    if not os.path.exists("my_plots/Mean/Linear/predicted_no2_Vs_epa_no2" ):
        os.mkdir( "my_plots/Mean/Linear/predicted_no2_Vs_epa_no2" )
        os.mkdir( "my_plots/Mean/Linear/predicted_no2_Vs_epa_no2/round1" )
        os.mkdir( "my_plots/Mean/Linear/predicted_no2_Vs_epa_no2/round2" )
        os.mkdir( "my_plots/Mean/Linear/predicted_no2_Vs_epa_no2/round3" )

    test = linear_time_step( timeStep, round, location, board )
    print("plotting round " + str(round) + " " + location + " Board # " + str(board))
    plt.scatter((test[PREDICT])[:, NO2], (test[ACTUAL])[:, NO2], s=.7)
    plt.xlabel('predicted no2')
    plt.ylabel('actual no2')
    plt.plot( range(0, 100), range(0, 100) )
    plt.xlim( 0, 90 )
    plt.ylim( 0, 90 ) 
    plt.title( "actual no2 Vs. predicted no2: " + str(location) + " board " + str(board) + " time step " + str(timeStep) ) 
    plt.savefig("my_plots/Mean/Linear/predicted_no2_Vs_epa_no2/round" + str(round) + "/" + str(round) + "-" + location + "-" + str(board) + "-" + str(timeStep) + ".png") 
    plt.clf()
    plt.scatter((test[PREDICT])[:, O3], (test[ACTUAL])[:, O3], s=.7)
    plt.xlabel('predicted o3')
    plt.ylabel('actual o3')
    plt.plot( range(0, 100), range(0, 100) )
    plt.xlim( 0, 90 )
    plt.ylim( 0, 90 )

    plt.title( "actual o3 Vs. predicted o3: " + str(location) + " board " + str(board) + " time step " + str(timeStep) )
    plt.savefig("my_plots/Mean/Linear/predicted_o3_Vs_epa_o3/round" + str(round) + "/" + str(round) + "-" + location + "-" + str(board) + "-" + str(timeStep) + ".png") 
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



