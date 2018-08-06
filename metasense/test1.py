import metasense
import data
from data import load5Second
import matching_epa
from matching_epa import match5Second
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#data = load5Second(1, 'donovan', 17)
#train = data[0]
#test = data[1]
#epa = data[2]
#print(epa)
#train = train.sort_values(by=['Ts'])
#print(train)
#plt.scatter(train.index.values[1 : 10], (train['no2'])[1 : 10], s = 0.5)
#plt.xlabel('time (5 seconds)')
#plt.ylabel('o3 (Active - Working Voltage)')  
#plt.savefig("my_plots/5second.png")
#plt.clf()
#data = match5Second( 2, 'donovan', 15)
#data = match5Second( 2, 'donovan', 18) #TODO
#data = match5Second( 2, 'donovan', 20)
#print("el cajon, round 1, board 12")
#data = match5Second( 1, 'elcajon', 12)
#print("el cajon, round 1, board 13")
#data = match5Second( 1, 'elcajon', 13)
#print("el cajon, round 3, board 15")
#data = match5Second( 3, 'elcajon', 15)
print("shafter, round 3, board 21")
data = match5Second( 3, 'shafter', 21)

