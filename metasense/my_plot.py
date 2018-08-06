import metasense
import data
import os, sys
from data import load
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def my_plot(x_axis, y_axis):
    if not os.path.exists("my_plots/" + y_axis + "Vs" + x_axis):
        os.mkdir( "my_plots/" + y_axis + "Vs" + x_axis)
        os.mkdir( "my_plots/" + y_axis + "Vs" + x_axis + "/round1")
        os.mkdir( "my_plots/" + y_axis + "Vs" + x_axis + "/round2")
        os.mkdir( "my_plots/" + y_axis + "Vs" + x_axis + "/round3") 
    #Round 1 donovan sensor 17
    data = load(1, 'donovan', 17)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round1/1-donovan-17.png") 
    plt.clf()

    #Round 1 donovan sensor 19
    data = load(1, 'donovan', 19)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis) 
    plt.title( y_axis + " vs. " + x_axis + ": Round 1 donovan")
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round1/1-donovan-19.png") 
    plt.clf()

    #Round 1 donovan sensor 21
    data = load(1, 'donovan', 21)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis) 
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round1/1-donovan-21.png")  
    plt.clf()

    #Round 1 el cajon sensor 11
    data = load(1, 'elcajon', 11)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round1/1-elcajon-11.png")  
    plt.clf()

    #Round 1 el cajon sensor 12
    data = load(1, 'elcajon', 12)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round1/1-elcajon-12.png")   
    plt.clf()

    #Round 1 el cajon sensor 13
    data = load(1, 'elcajon', 13)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round1/1-elcajon-13.png")   
    plt.clf()

    #Round 1 shafter sensor 15
    data = load(1, 'shafter', 15)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round1/1-shafter-15.png")   
    plt.clf()

    #Round 1 shafter sensor 18
    data = load(1, 'shafter', 18)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round1/1-shafter-18.png")   
    plt.clf()

    #Round 2 donovan sensor 15
    data = load(2, 'donovan', 15)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round2/2-donovan-15.png")   
    plt.clf()


    #Round 2 donovan sensor 18
    data = load(2, 'donovan', 18)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round2/2-donovan-18.png")   
    plt.clf()

    #Round 2 donovan sensor 20
    data = load(2, 'donovan', 20)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round2/2-donovan-20.png")   
    plt.clf()

    #Round 2 el cajon sensor 17
    data = load(2, 'elcajon', 17)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round2/2-elcajon-17.png")   
    plt.clf()

    #Round 2 el cajon sensor 19
    data = load(2, 'elcajon', 19)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round2/2-elcajon-19.png")   
    plt.clf()

    #Round 2 el cajon sensor 21
    data = load(2, 'elcajon', 21)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round2/2-elcajon-21.png")   
    plt.clf()

    #Round 2 shafter sensor 11
    data = load(2, 'shafter', 11)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round2/2-shafter-11.png")   
    plt.clf()

    #Round 2 shafter sensor 12
    data = load(2, 'shafter', 12)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round2/2-shafter-12.png")   
    plt.clf()
 
    #Round 2 shafter sensor 13
    data = load(2, 'shafter', 13)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round2/2-shafter-13.png")   
    plt.clf()

    #Round 3 donovan sensor 11
    data = load(3, 'donovan', 11)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round3/3-donovan-11.png")   
    plt.clf()

    #Round 3 donovan sensor 12
    data = load(3, 'donovan', 12)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round3/3-donovan-12.png")   
    plt.clf()

    #Round 3 donovan sensor 13
    data = load(3, 'donovan', 13)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round3/3-donovan-13.png")   
    plt.clf()

    #Round 3 el cajon sensor 15
    data = load(3, 'elcajon', 15)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round3/3-elcajon-15.png")   
    plt.clf()

    #Round 3 el cajon sensor 18
    data = load(3, 'elcajon', 18)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round3/3-elcajon-18.png")   
    plt.clf()

    #Round 3 el cajon sensor 20
    data = load(3, 'elcajon', 20)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round3/3-elcajon-20.png")   
    plt.clf()

    #Round 3 shafter sensor 17
    data = load(3, 'shafter', 17)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round3/3-shafter-17.png")   
    plt.clf()

    #Round 3 shafter sensor 19
    data = load(3, 'shafter', 19)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round3/3-shafter-19.png")   
    plt.clf()

    #Round 3 shafter sensor 21
    data = load(3, 'shafter', 21)
    train = data[0]
    test = data [1]
    trainY = train[y_axis]
    trainY = trainY.sort_index()
    y = trainY.values
    if x_axis == 'time':
        x = np.arange(0, len(y), 1.0)
    else:
        trainX = train[x_axis]
        trainX = trainX.sort_index()
        x = trainX.values
    plt.scatter(x, y, s=.7)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title( y_axis + " vs. " + x_axis )
    plt.savefig("my_plots/" + y_axis + "Vs" + x_axis + "/round3/3-shafter-21.png")   
    plt.clf()


#Running the codes
#time
print("Plotting temperature vs time ...")
my_plot('time', 'temperature')
print("Plotting no2 vs time ...")
my_plot('time', 'no2')
print("Plotting o3 vs time ...")
my_plot('time', 'o3')
print("Plotting epa-no2 vs time ...")
my_plot('time', 'epa-no2')
print("Plotting epa-03 vs time ...")
my_plot('time', 'epa-o3')
print("Plotting absolute humidity vs time ...")
my_plot('time', 'absolute-humidity')
print("Plotting co vs time ...")
my_plot('time', 'co')
#humidity
print("Plotting no2 vs humidity ...")
my_plot('absolute-humidity', 'no2')
print("Plotting o3 vs humidity ...")
my_plot('absolute-humidity', 'o3')
print("Plotting co vs humidity ...")
my_plot('absolute-humidity', 'co')
print("Plotting epa-no2 vs humidity ...")
my_plot('absolute-humidity', 'epa-no2')
print("Plotting epa-o3 vs humidity ...")
my_plot('absolute-humidity', 'epa-o3')
print("Plotting temperature vs humidity ...")
my_plot('absolute-humidity', 'temperature')
#temperature
print("Plotting no2 vs temeperature ...")
my_plot('temperature', 'no2')
print("Plotting o3 vs temeperature ...")
my_plot('temperature', 'o3')
print("Plotting co vs temeperature ...")
my_plot('temperature', 'co')
print("Plotting epa-no2 vs temeperature ...")
my_plot('temperature', 'epa-no2')
print("Plotting epa-o3 vs temeperature ...")
my_plot('temperature', 'epa-o3')
#epa-no2
print("Plotting no2 vs epa-no2 ...")
my_plot('epa-no2', 'no2')
print("Plotting o3 vs epa-no2 ...")
my_plot('epa-no2', 'o3')
print("Plotting co vs epa-no2 ...")
my_plot('epa-no2', 'co')
print("Plotting epa-o3 vs epa-no2 ...")
my_plot('epa-no2', 'epa-o3')
#epa-o3
print("Plotting no2 vs epa-o3 ...")
my_plot('epa-o3', 'no2')
print("Plotting o3 vs epa-o3 ...")
my_plot('epa-o3', 'o3')
print("Plotting co vs epa-o3 ...")
my_plot('epa-o3', 'co')
#no2
print("Plotting o3 vs no2 ...")
my_plot('no2', 'o3')
print("Plotting co vs no2 ...")
my_plot('no2', 'co')
#o3
print("Plotting co vs o3 ...")
my_plot('o3', 'co')
