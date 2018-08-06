import metasense
import data
import os, sys
from data import load5Second
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def plot_5_second( round, location, board ):
    if not os.path.exists("my_plots/5_second"):
        os.mkdir( "my_plots/5_second" )
    
    if not os.path.exists("my_plots/5_second/Difference"):
        os.mkdir( "my_plots/5_second/Difference" )

    if not os.path.exists("my_plots/5_second/Working"):
        os.mkdir( "my_plots/5_second/Working" )
    
    if not os.path.exists("my_plots/5_second/Auxillary"):
        os.mkdir( "my_plots/5_second/Auxillary" )
      
    if not os.path.exists("my_plots/5_second/Difference/o3_Vs_time" ):
        os.mkdir( "my_plots/5_second/Difference/o3_Vs_time" )
        os.mkdir( "my_plots/5_second/Difference/o3_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Difference/o3_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Difference/o3_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Difference/o3_Vs_time/round4" )

    if not os.path.exists("my_plots/5_second/Difference/no2_Vs_time" ):
        os.mkdir( "my_plots/5_second/Difference/no2_Vs_time" )
        os.mkdir( "my_plots/5_second/Difference/no2_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Difference/no2_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Difference/no2_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Difference/no2_Vs_time/round4" )

    if not os.path.exists("my_plots/5_second/Working/o3_Vs_time" ):
        os.mkdir( "my_plots/5_second/Working/o3_Vs_time" )
        os.mkdir( "my_plots/5_second/Working/o3_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Working/o3_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Working/o3_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Working/o3_Vs_time/round4" )

    if not os.path.exists("my_plots/5_second/Working/no2_Vs_time" ):
        os.mkdir( "my_plots/5_second/Working/no2_Vs_time" )
        os.mkdir( "my_plots/5_second/Working/no2_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Working/no2_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Working/no2_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Working/no2_Vs_time/round4" )

    if not os.path.exists("my_plots/5_second/Auxillary/o3_Vs_time" ):
        os.mkdir( "my_plots/5_second/Auxillary/o3_Vs_time" )
        os.mkdir( "my_plots/5_second/Auxillary/o3_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Auxillary/o3_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Auxillary/o3_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Auxillary/o3_Vs_time/round4" )

    if not os.path.exists("my_plots/5_second/Auxillary/no2_Vs_time" ):
        os.mkdir( "my_plots/5_second/Auxillary/no2_Vs_time" )
        os.mkdir( "my_plots/5_second/Auxillary/no2_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Auxillary/no2_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Auxillary/no2_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Auxillary/no2_Vs_time/round4" )



    data = load5Second(round, location, board)
    train = data[0]
    train = train.sort_values(by=['Ts'])  
    print("plotting " + str(round) + " " + location + " " + str(board) + " ...")
    plt.scatter(train.index.values, (train['no2']), s = 0.5)
    plt.xlabel('time (5 seconds)')
    plt.ylabel('no2 (mV)')  
    plt.title("5 Second Difference no2: Round " + str(round) + " " + location + " Board #" + str(board))
    plt.savefig("my_plots/5_second/Difference/no2_Vs_time/round" + str(round) + "/" + location + str(board) + ".png")
    plt.clf()
    plt.scatter(train.index.values, (train['o3']), s = 0.5)
    plt.xlabel('time (5 seconds)')
    plt.ylabel('o3 (mV)')  
    plt.title("5 Second Difference o3: Round " + str(round) + " " + location + " Board #" + str(board)) 
    plt.savefig("my_plots/5_second/Difference/o3_Vs_time/round" + str(round) + "/" + location + str(board) + ".png")
    plt.clf()
    plt.scatter(train.index.values, (train['No2WmV']), s = 0.5)
    plt.xlabel('time (5 seconds)')
    plt.ylabel('no2 (mV)')  
    plt.title("5 Second Working no2: Round " + str(round) + " " + location + " Board #" + str(board))
    plt.savefig("my_plots/5_second/Working/no2_Vs_time/round" + str(round) + "/" + location + str(board) + ".png")
    plt.clf()
    plt.scatter(train.index.values, (train['OxWmV']), s = 0.5)
    plt.xlabel('time (5 seconds)')
    plt.ylabel('o3 (mV)')  
    plt.title("5 Second Working o3: Round " + str(round) + " " + location + " Board #" + str(board))
    plt.savefig("my_plots/5_second/Working/o3_Vs_time/round" + str(round) + "/" + location + str(board) + ".png")
    plt.clf()
    plt.scatter(train.index.values, (train['No2AmV']), s = 0.5)
    plt.xlabel('time (5 seconds)')
    plt.ylabel('no2 (mV)')  
    plt.title("5 Second Auxillary no2: Round " + str(round) + " " + location + " Board #" + str(board))
    plt.savefig("my_plots/5_second/Auxillary/no2_Vs_time/round" + str(round) + "/" + location + str(board) + ".png")
    plt.clf()
    plt.scatter(train.index.values, (train['OxAmV']), s = 0.5)
    plt.xlabel('time (5 seconds)')
    plt.ylabel('o3 (mV)')  
    plt.title("5 Second Auxillary no2: Round " + str(round) + " " + location + " Board #" + str(board))
    plt.savefig("my_plots/5_second/Auxillary/o3_Vs_time/round" + str(round) + "/" + location + str(board) + ".png")
    plt.clf()


def plot_5_second_window( round, location, board, start, end ):
    if not os.path.exists("my_plots/5_second"):
        os.mkdir( "my_plots/5_second" )
    
    if not os.path.exists("my_plots/5_second/Difference"):
        os.mkdir( "my_plots/5_second/Difference" )

    if not os.path.exists("my_plots/5_second/Working"):
        os.mkdir( "my_plots/5_second/Working" )
    
    if not os.path.exists("my_plots/5_second/Auxillary"):
        os.mkdir( "my_plots/5_second/Auxillary" )
      
    if not os.path.exists("my_plots/5_second/Difference/o3_Vs_time" ):
        os.mkdir( "my_plots/5_second/Difference/o3_Vs_time" )
        os.mkdir( "my_plots/5_second/Difference/o3_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Difference/o3_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Difference/o3_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Difference/o3_Vs_time/round4" )

    if not os.path.exists("my_plots/5_second/Difference/no2_Vs_time" ):
        os.mkdir( "my_plots/5_second/Difference/no2_Vs_time" )
        os.mkdir( "my_plots/5_second/Difference/no2_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Difference/no2_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Difference/no2_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Difference/no2_Vs_time/round4" )

    if not os.path.exists("my_plots/5_second/Working/o3_Vs_time" ):
        os.mkdir( "my_plots/5_second/Working/o3_Vs_time" )
        os.mkdir( "my_plots/5_second/Working/o3_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Working/o3_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Working/o3_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Working/o3_Vs_time/round4" )

    if not os.path.exists("my_plots/5_second/Working/no2_Vs_time" ):
        os.mkdir( "my_plots/5_second/Working/no2_Vs_time" )
        os.mkdir( "my_plots/5_second/Working/no2_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Working/no2_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Working/no2_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Working/no2_Vs_time/round4" )

    if not os.path.exists("my_plots/5_second/Auxillary/o3_Vs_time" ):
        os.mkdir( "my_plots/5_second/Auxillary/o3_Vs_time" )
        os.mkdir( "my_plots/5_second/Auxillary/o3_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Auxillary/o3_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Auxillary/o3_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Auxillary/o3_Vs_time/round4" )

    if not os.path.exists("my_plots/5_second/Auxillary/no2_Vs_time" ):
        os.mkdir( "my_plots/5_second/Auxillary/no2_Vs_time" )
        os.mkdir( "my_plots/5_second/Auxillary/no2_Vs_time/round1" )
        os.mkdir( "my_plots/5_second/Auxillary/no2_Vs_time/round2" )
        os.mkdir( "my_plots/5_second/Auxillary/no2_Vs_time/round3" )
        os.mkdir( "my_plots/5_second/Auxillary/no2_Vs_time/round4" )



    data = load5Second(round, location, board)
    train = data[0]
    train = train.sort_values(by=['Ts']) 
    print("plotting " + str(round) + " " + location + " " + str(board) + " ...")
    plt.plot(train.index.values[start : end], (train['no2'])[start : end])
    plt.xlabel('time (5 seconds)')
    plt.ylabel('no2 (mV)')  
    plt.title("5 Second Difference no2: Round " + str(round) + " " + location + " Board #" + str(board) + " " + str(start) + " to " + str(end))
    plt.savefig("my_plots/5_second/Difference/no2_Vs_time/round" + str(round) + "/" + location + str(board) + "_" + str(start) + "_to_" + str(end) + ".png")
    plt.clf()
    plt.plot(train.index.values[start : end], (train['o3'])[start : end])
    plt.xlabel('time (5 seconds)')
    plt.ylabel('o3 (mV)')  
    plt.title("5 Second Difference o3: Round " + str(round) + " " + location + " Board #" + str(board) + " " + str(start) + " to " + str(end)) 
    plt.savefig("my_plots/5_second/Difference/o3_Vs_time/round" + str(round) + "/" + location + str(board) + "_" + str(start) + "_to_" + str(end) + ".png")
    plt.clf()
    plt.plot(train.index.values[start : end], (train['No2WmV'])[start : end])
    plt.xlabel('time (5 seconds)')
    plt.ylabel('no2 (mV)')  
    plt.title("5 Second Working no2: Round " + str(round) + " " + location + " Board #" + str(board) + " " + str(start) + " to " + str(end))
    plt.savefig("my_plots/5_second/Working/no2_Vs_time/round" + str(round) + "/" + location + str(board) + "_" + str(start) + "_to_" + str(end) + ".png")
    plt.clf()
    plt.plot(train.index.values[start : end], (train['OxWmV'])[start : end])
    plt.xlabel('time (5 seconds)')
    plt.ylabel('o3 (mV)')  
    plt.title("5 Second Working o3: Round " + str(round) + " " + location + " Board #" + str(board) + " " + str(start) + " to " + str(end))
    plt.savefig("my_plots/5_second/Working/o3_Vs_time/round" + str(round) + "/" + location + str(board) + "_" + str(start) + "_to_" + str(end) +  ".png")
    plt.clf()
    plt.plot(train.index.values[start : end], (train['No2AmV'])[start : end])
    plt.xlabel('time (5 seconds)')
    plt.ylabel('no2 (mV)')  
    plt.title("5 Second Auxillary no2: Round " + str(round) + " " + location + " Board #" + str(board) + " " + str(start) + " to " + str(end))
    plt.savefig("my_plots/5_second/Auxillary/no2_Vs_time/round" + str(round) + "/" + location + str(board) + "_" + str(start) + "_to_" + str(end) + ".png")
    plt.clf()
    plt.plot(train.index.values[start : end], (train['OxAmV'])[start : end])
    plt.xlabel('time (5 seconds)')
    plt.ylabel('o3 (mV)')  
    plt.title("5 Second Auxillary no2: Round " + str(round) + " " + location + " Board #" + str(board) + " " + str(start) + " to " + str(end))
    plt.savefig("my_plots/5_second/Auxillary/o3_Vs_time/round" + str(round) + "/" + location + str(board) + "_" + str(start) + "_to_" + str(end) + ".png")
    plt.clf()



   

   

plot_5_second_window( 1, 'donovan', 19, 10, 100)
#plot_5_second( 1, 'donovan', 17 )
#plot_5_second( 1, 'donovan', 19 )
#plot_5_second( 1, 'donovan', 21 )
#plot_5_second( 2, 'donovan', 15 )
#plot_5_second( 2, 'donovan', 18 )
#plot_5_second( 2, 'donovan', 20 )
#plot_5_second( 3, 'donovan', 11 )
#plot_5_second( 3, 'donovan', 12 )
#plot_5_second( 3, 'donovan', 13 )
#plot_5_second( 4, 'donovan', 17 )
#plot_5_second( 4, 'donovan', 19 )
#plot_5_second( 4, 'donovan', 21 )
#plot_5_second( 1, 'elcajon', 11 )
#plot_5_second( 1, 'elcajon', 12 )
#plot_5_second( 1, 'elcajon', 13 )
#plot_5_second( 2, 'elcajon', 17 )
#plot_5_second( 2, 'elcajon', 19 )
#plot_5_second( 2, 'elcajon', 21 )
#plot_5_second( 3, 'elcajon', 15 )
#plot_5_second( 3, 'elcajon', 18 )
#plot_5_second( 3, 'elcajon', 20 )
#plot_5_second( 4, 'elcajon', 11 )
#plot_5_second( 4, 'elcajon', 12 )
#plot_5_second( 4, 'elcajon', 13 )
#plot_5_second( 1, 'shafter', 15 )
#plot_5_second( 1, 'shafter', 18 )
#plot_5_second( 1, 'shafter', 20 )
#plot_5_second( 2, 'shafter', 11 )
#plot_5_second( 2, 'shafter', 12 )
#plot_5_second( 2, 'shafter', 13 )
#plot_5_second( 3, 'shafter', 17 )
#plot_5_second( 3, 'shafter', 19 )
#plot_5_second( 3, 'shafter', 21 )
#plot_5_second( 4, 'shafter', 15 )
#plot_5_second( 4, 'shafter', 18 )
#plot_5_second( 4, 'shafter', 20 )



