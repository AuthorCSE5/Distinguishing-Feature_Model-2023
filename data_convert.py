import numpy as np
import random
import math
from scipy.special import comb
import time
import pandas as pd

'''
Dota = np.array(pd.read_csv('tennis_new.csv'))

num_items = 742

print("Dota = ", Dota)
Dota2 = Dota[num_items+1:,0:4]
Dota2 = Dota2.astype('int')
print("Dota = ", Dota2)
dt_size = Dota2.shape[0]
print(dt_size)
print(type(Dota2))
'''

def convert(data_name, num_items, data_size, dt_size):

    print("data_size = ", data_size)

    print("Converting Data...")

    dt_final = np.zeros((int(data_size), 4))

    count1 = sum((data_name[:,2]))
    count2 = sum((data_name[:,3]))
    count = count1 + count2
    dt = np.zeros((1, 4))
    w = 0
    
    for k in range(dt_size):
        #print(data_name[k][2],data_name[k][3])
        c1 = data_name[k][2]
        c2 = data_name[k][3]
        c3 = c1 + c2
        
        
        for count in range(c1):
            #print("c1 = ", c1)
            dt[0][0] = data_name[k][0]
            dt[0][1] = data_name[k][1]
            dt[0][2] = 1
            dt[0][3] = 0
            #print(dt)
            dt_final[w][:] = dt
            w += 1
            #print(dt_final)
            c1 -= 1
        for count in range(c2):
            #print("c2 = ", c2)
            dt[0][0] = data_name[k][0]
            dt[0][1] = data_name[k][1]
            dt[0][2] = 0
            dt[0][3] = 1
            #print(dt)
            dt_final[w][:] = dt
            w += 1
            #print(dt_final)
            c2 -= 1
            

    #dt_final = np.array(dt_final)
    print("dt_final = ", dt_final)

    return dt_final

#Dota = data_convert(Dota2)
#print("dt_final = ", Dota)
