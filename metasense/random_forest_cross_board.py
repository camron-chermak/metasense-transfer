import metasense
import data
import os, sys
from data import loadMatrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor


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


def random_forest_cross_board_time_step( timeStep, round, location, board1, board2, testBoard):
    TIME_STEP = timeStep
    rf = RandomForestRegressor()
    data1 = loadMatrix( TIME_STEP, round, location, board1) 
    if board2 != 0:
        data2 = loadMatrix( TIME_STEP, round, location, board2)
    cross = loadMatrix( TIME_STEP, round, location, testBoard)
    train = data1[0]
    train = np.concatenate((train, data1[1]), axis=0 )
    if board2 != 0:
        train = np.concatenate((train, data2[0]), axis=0 )
        train = np.concatenate((train, data2[1]), axis=0 )
    test = cross[0]
    test = np.concatenate((test, cross[1]), axis=0 )
    trainX = train[TIME_STEP::, NO2_INDEX]
    trainX = np.expand_dims(trainX, axis=1)
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, O3_INDEX], axis=1)), axis=1)
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, CO_INDEX], axis=1)), axis=1)
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, TEMP_INDEX], axis=1)), axis=1)
    trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, HUM_INDEX], axis=1)), axis=1)
    for index in range(TIME_STEP):
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, NO2_INDEX + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, O3_INDEX + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, CO_INDEX + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, TEMP_INDEX + (index + 1)*8], axis=1)), axis=1)
        trainX = np.concatenate( (trainX, np.expand_dims(train[TIME_STEP::, HUM_INDEX + (index + 1)*8], axis=1)), axis=1)
    trainY = train[TIME_STEP::,EPA_NO2_INDEX]
    trainY = np.expand_dims(trainY, axis=1)
    trainY = np.concatenate( (trainY, np.expand_dims(train[TIME_STEP::, EPA_O3_INDEX], axis=1)), axis=1) 
    rf.fit(trainX, trainY)
    
    testX = test[TIME_STEP::, NO2_INDEX]
    testX = np.expand_dims(testX, axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, O3_INDEX], axis=1)), axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, CO_INDEX], axis=1)), axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, TEMP_INDEX], axis=1)), axis=1)
    testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, HUM_INDEX], axis=1)), axis=1)
    for index in range(TIME_STEP):
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, NO2_INDEX + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, O3_INDEX + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, CO_INDEX + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, TEMP_INDEX + (index + 1)*8], axis=1)), axis=1)
        testX = np.concatenate( (testX, np.expand_dims(test[TIME_STEP::, HUM_INDEX + (index + 1)*8], axis=1)), axis=1)
    testY = test[TIME_STEP::, EPA_NO2_INDEX]
    testY = np.expand_dims(testY, axis=1)
    testY = np.concatenate( (testY, np.expand_dims(test[TIME_STEP::, EPA_O3_INDEX], axis=1)), axis=1) 
    y_predict = rf.predict(testX)
    return y_predict, testY

def plot_random_forest_cross_board_time_step( timeStep, round, location, board1, board2, testBoard):
    if not os.path.exists("my_plots/Mean"):
        os.mkdir( "my_plots/Mean" )
    
    if not os.path.exists("my_plots/Mean/Random_Forest"):
        os.mkdir( "my_plots/Mean/Random_Forest" )

    if not os.path.exists("my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3" ):
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3/round1" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3/round2" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3/round3" )
      
    if not os.path.exists( "my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Board" ):
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Board" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Board/donovan" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Board/elcajon" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Board/shafter" )

    if not os.path.exists("my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2" ):
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2/round1" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2/round2" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2/round3" )

    if not os.path.exists( "my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Board" ): 
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Board" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Board/donovan" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Board/elcajon" )
        os.mkdir( "my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Board/shafter" )
    

    test = random_forest_cross_board_time_step( timeStep, round, location, board1, board2, testBoard )
    print("plotting " + location + " Board # " + str(testBoard))

    plt.scatter((test[PREDICT])[:, NO2], (test[ACTUAL])[:, NO2], s=.7)
    plt.xlabel('predicted no2')
    plt.ylabel('actual no2')
    plt.plot( range(0, 100), range(0, 100) )
    plt.xlim( 0, 90 )
    plt.ylim( 0, 90 ) 
    plt.title( "no2: " + location + " to test " + str(testBoard) ) 
    plt.savefig("my_plots/Mean/Random_Forest/predicted_no2_Vs_epa_no2/Cross_Board/" + location + "/" + location + "_" + str(round) + "_" + str(testBoard) + ".png") 
    plt.clf()

    plt.scatter((test[PREDICT])[:, O3], (test[ACTUAL])[:, O3], s=.7)
    plt.xlabel('predicted o3')
    plt.ylabel('actual o3')
    plt.plot( range(0, 100), range(0, 100) )
    plt.xlim( 0, 90 )
    plt.ylim( 0, 90 )  
    plt.title( "o3: " + location + " to test " + str(testBoard) ) 
    plt.savefig("my_plots/Mean/Random_Forest/predicted_o3_Vs_epa_o3/Cross_Board/" + location + "/" + location + "_" + str(round) + "_" + str(testBoard) + ".png") 
    plt.clf()


# does not include board 15, 19; 20 for round 1
def plotting_cross_board(time):
    print("Donovan round 1")
    plot_random_forest_cross_board_time_step(time, 1, 'donovan', 17, 0, 21) 
    plot_random_forest_cross_board_time_step(time, 1, 'donovan', 21, 0, 17)

    print("Donovan round 2")
    plot_random_forest_cross_board_time_step(time, 2, 'donovan', 18, 0, 20) 
    plot_random_forest_cross_board_time_step(time, 2, 'donovan', 20, 0, 18)
 
    print("Donovan round 3")
    plot_random_forest_cross_board_time_step(time, 3, 'donovan', 11, 12, 13) 
    plot_random_forest_cross_board_time_step(time, 3, 'donovan', 11, 13, 12)
    plot_random_forest_cross_board_time_step(time, 3, 'donovan', 12, 13, 11) 

    print("El Cajon round 1")
    plot_random_forest_cross_board_time_step(time, 1, 'elcajon', 11, 12, 13) 
    plot_random_forest_cross_board_time_step(time, 1, 'elcajon', 11, 13, 12)
    plot_random_forest_cross_board_time_step(time, 1, 'elcajon', 12, 13, 11) 

    print("El Cajon round 2")
    plot_random_forest_cross_board_time_step(time, 2, 'elcajon', 17, 0, 21) 
    plot_random_forest_cross_board_time_step(time, 2, 'elcajon', 21, 0, 17)

    print("El Cajon round 3")
    plot_random_forest_cross_board_time_step(time, 3, 'elcajon', 18, 0, 20) 
    plot_random_forest_cross_board_time_step(time, 3, 'elcajon', 20, 0, 18)

    print("Shafter round 2")
    plot_random_forest_cross_board_time_step(time, 2, 'shafter', 11, 12, 13) 
    plot_random_forest_cross_board_time_step(time, 2, 'shafter', 11, 13, 12)
    plot_random_forest_cross_board_time_step(time, 2, 'shafter', 12, 13, 11) 

    print("Shafter round 3")
    plot_random_forest_cross_board_time_step(time, 3, 'shafter', 17, 0, 21) 
    plot_random_forest_cross_board_time_step(time, 3, 'shafter', 21, 0, 17)


plotting_cross_board(0)
