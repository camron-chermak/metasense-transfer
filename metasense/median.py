import numpy as np
import os, sys
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def match_median(round, location, board_id, seed=0):
    if not os.path.exists("Median_Data"):
        os.mkdir("Median_Data")

    if not os.path.exists("Median_Data/" + location):
        os.mkdir( "Median_Data/" + location )
        os.mkdir( "Median_Data/" + location + "/round1")
        os.mkdir( "Median_Data/" + location + "/round2")
        os.mkdir( "Median_Data/" + location + "/round3") 
        os.mkdir( "Median_Data/" + location + "/round4")

    print( "matching " + location + " round " + str(round) + " Board # " + str(board_id) )
    path = "match_5Second/" + location + "/Round" + str(round) + "/match_round_" + str(round) + "_" + location + "_" + str(board_id) + ".csv"
    data = pd.read_csv(path, parse_dates=True)
    median = 0.0
    myNo2 = np.array(data['no2'][0])
    myo3 = np.array(data['o3'][0]) 
    data['median-no2'] = -1.0
    data['median-o3'] = -1.0
    for index in range(len(data) - 1):
        curr = index + 1
        print(( curr/(len(data))) * 100) 
        if data['Time'][curr] == data['Time'][curr - 1]: 
            myNo2 = np.append(myNo2, data['no2'][curr])
            myo3 = np.append(myo3, data['o3'][curr])
        else: 
            data.at[(curr - 1), 'median-no2'] = np.median(myNo2)
            data.at[(curr - 1), 'median-o3'] = np.median(myo3)
            myNo2 = np.array(data['no2'][curr])
            myo3 = np.array(data['o3'][curr])

    data = data[data['median-no2'] != -1.0]
    data = data[data['median-o3'] != -1.0]
    data['no2'] = data['median-no2']
    data['o3'] = data['median-o3']
    data = data.drop(['median-no2', 'median-o3'], axis=1)
    outPath = "Median_Data/" + location + "/round" + str(round) + "/" + location + "_" + str(board_id) + ".csv"
    data.to_csv(outPath)
    train, test = train_test_split(data, test_size=0.2, random_state=seed)
    return train, test


def plot_median(round, location, board_id, seed=0):

    path = "Median_Data/" + location + "/round" + str(round) + "/" + location + "_" + str(board_id) + ".csv" 
    data = pd.read_csv(path, parse_dates=True)

    if not os.path.exists("my_plots/Median"):
        os.mkdir("my_plots/Median")

    if not os.path.exists("my_plots/Median/no2_Vs_time" ):
        os.mkdir( "my_plots/Median/no2_Vs_time" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round1" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round1/donovan" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round1/elcajon" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round1/shafter" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round2" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round2/donovan" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round2/elcajon" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round2/shafter" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round3" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round3/donovan" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round3/elcajon" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round3/shafter" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round4" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round4/donovan" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round4/elcajon" )
        os.mkdir( "my_plots/Median/no2_Vs_time/round4/shafter" )
        
    if not os.path.exists("my_plots/Median/o3_Vs_time" ):
        os.mkdir( "my_plots/Median/o3_Vs_time" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round1" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round1/donovan" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round1/shafter" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round1/elcajon" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round2" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round2/donovan" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round2/shafter" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round2/elcajon" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round3" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round3/donovan" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round3/shafter" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round3/elcajon" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round4" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round4/donovan" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round4/shafter" )
        os.mkdir( "my_plots/Median/o3_Vs_time/round4/elcajon" )
  
    if not os.path.exists("my_plots/Median/no2_Vs_epa-no2" ):
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round1" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round1/donovan" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round1/elcajon" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round1/shafter" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round2" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round2/donovan" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round2/elcajon" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round2/shafter" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round3" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round3/donovan" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round3/elcajon" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round3/shafter" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round4" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round4/donovan" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round4/elcajon" )
        os.mkdir( "my_plots/Median/no2_Vs_epa-no2/round4/shafter" )
      
    if not os.path.exists("my_plots/Median/o3_Vs_epa-o3" ):
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round1" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round1/donovan" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round1/elcajon" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round1/shafter" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round2" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round2/donovan" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round2/elcajon" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round2/shafter" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round3" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round3/donovan" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round3/elcajon" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round3/shafter" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round4" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round4/donovan" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round4/elcajon" )
        os.mkdir( "my_plots/Median/o3_Vs_epa-o3/round4/shafter" )
 
    print("plotting round " + str(round) + " " + location + " Board # " + str(board_id))
    plt.scatter(data.index.values, data['no2'], s=.7) 
    plt.xlabel('time (min)')
    plt.ylabel('no2 (mV)') 
    plt.title( "median no2 Vs time: " + str(location) + " board " + str(board_id) ) 
    plt.savefig("my_plots/Median/no2_Vs_time/"  + "/round" + str(round) + "/"  + location + "/" + location + "_" + str(board_id) + ".png") 
    plt.clf()
    plt.scatter(data.index.values, data['o3'], s=.7) 
    plt.xlabel('time (min)')
    plt.ylabel('o3 (mV)') 
    plt.title( "median o3 Vs time: " + str(location) + " board " + str(board_id) ) 
    plt.savefig("my_plots/Median/o3_Vs_time/"  + "/round" + str(round) + "/"  + location + "/" + location + "_" + str(board_id) + ".png") 
    plt.clf()
    plt.scatter(data['epa-no2'], data['no2'], s=.7) 
    plt.xlabel('epa no2')
    plt.ylabel('no2 (mV)') 
    plt.title( "median no2 Vs epa-no2: " + str(location) + " board " + str(board_id) ) 
    plt.savefig("my_plots/Median/no2_Vs_epa-no2/"  + "/round" + str(round) + "/"  + location + "/" + location + "_" + str(board_id) + ".png") 
    plt.clf()
    plt.scatter(data['epa-o3'], data['o3'], s=.7) 
    plt.xlabel('epa o3')
    plt.ylabel('o3 (mV)') 
    plt.title( "median o3 Vs epa-o3: " + str(location) + " board " + str(board_id) ) 
    plt.savefig("my_plots/Median/o3_Vs_epa-o3/"  + "/round" + str(round) + "/"  + location + "/" + location + "_" + str(board_id) + ".png") 
    plt.clf()


 
#plot_median(1, 'donovan', 17) #DONE
#plot_median(1, 'donovan', 19) #DONE
#plot_median(1, 'donovan', 21) #DONE
#plot_median(2, 'donovan', 15) #DONE
#plot_median(2, 'donovan', 18) #DONE
#plot_median(2, 'donovan', 20) #DONE
#plot_median(3, 'donovan', 11) #DONE
#plot_median(3, 'donovan', 12) #DONE
#plot_median(3, 'donovan', 13) #DONE
#plot_median(4, 'donovan', 17) #TODO
#plot_median(4, 'donovan', 19) #TODO
#plot_median(4, 'donovan', 21) #TODO
#plot_median(1, 'elcajon', 11) #DONE
#plot_median(1, 'elcajon', 12) #DONE
#plot_median(1, 'elcajon', 13) #DONE
#plot_median(2, 'elcajon', 17) #DONE
#plot_median(2, 'elcajon', 19) #DONE
#plot_median(2, 'elcajon', 21) #DONE
#plot_median(3, 'elcajon', 15) #DONE
#plot_median(3, 'elcajon', 18) #DONE
#plot_median(3, 'elcajon', 20) #DONE
#plot_median(4, 'elcajon', 11) #TODO
#plot_median(4, 'elcajon', 12) #TODO
#plot_median(4, 'elcajon', 13) #TODO
#plot_median(1, 'shafter', 15) #DONE
#plot_median(1, 'shafter', 18) #DONE
#plot_median(1, 'shafter', 20) #DONE
#plot_median(2, 'shafter', 11) #DONE
#plot_median(2, 'shafter', 12) #DONE
#plot_median(2, 'shafter', 13) #DONE
#plot_median(3, 'shafter', 17) #DONE
#plot_median(3, 'shafter', 19) #DONE
#plot_median(3, 'shafter', 21) #DONE
#plot_median(4, 'shafter', 15) #TODO
#plot_median(4, 'shafter', 18) #TODO
#plot_median(4, 'shafter', 20) #TODO





#data = match_median(1, 'donovan', 17) #DONE
#data = match_median(1, 'donovan', 19) #DONE
#data = match_median(1, 'donovan', 21) #DONE
#data = match_median(2, 'donovan', 15) #DONE
#data = match_median(2, 'donovan', 18) #DONE
#data = match_median(2, 'donovan', 20) #DONE
#data = match_median(3, 'donovan', 11) #DONE
#data = match_median(3, 'donovan', 12) #DONE
#data = match_median(3, 'donovan', 13) #DONE
#data = match_median(4, 'donovan', 17) #TODO
#data = match_median(4, 'donovan', 19) #TODO
#data = match_median(4, 'donovan', 21) #TODO
#data = match_median(1, 'elcajon', 11) #DONE
#data = match_median(1, 'elcajon', 12) #DONE
#data = match_median(1, 'elcajon', 13) #DONE
#data = match_median(2, 'elcajon', 17) #DONE
#data = match_median(2, 'elcajon', 19) #DONE
#data = match_median(2, 'elcajon', 21) #DONE
#data = match_median(3, 'elcajon', 15) #DONE
#data = match_median(3, 'elcajon', 18) #DONE
#data = match_median(3, 'elcajon', 20) #DONE
#data = match_median(4, 'elcajon', 11) #TODO
#data = match_median(4, 'elcajon', 12) #TODO
#data = match_median(4, 'elcajon', 13) #TODO
#data = match_median(1, 'shafter', 15) #DONE
#data = match_median(1, 'shafter', 18) #DONE
#data = match_median(1, 'shafter', 20) #DONE
#data = match_median(2, 'shafter', 11) #DONE
#data = match_median(2, 'shafter', 12) #DONE
#data = match_median(2, 'shafter', 13) #DONE
#data = match_median(3, 'shafter', 17) #DONE
#data = match_median(3, 'shafter', 19) #DONE
#data = match_median(3, 'shafter', 21) #DONE
#data = match_median(4, 'shafter', 15) #TODO
#data = match_median(4, 'shafter', 18) #TODO
#data = match_median(4, 'shafter', 20) #TODO










