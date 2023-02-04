import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from scipy.special import comb
import pandas as pd
from sklearn.model_selection import train_test_split

import Real_Data
import create_data2
import create_data3


Dota = np.array(pd.read_csv('DOTA.csv')) #also used for jester.csv
#Dota = np.array(pd.read_csv('jester.csv'))
#Dota = np.array(pd.read_csv('WoL_Starcraft.csv'))
#Dota = np.array(pd.read_csv('movielens_100k.csv'))
#Dota = np.array(pd.read_csv('HotS_Starcraft.csv'))
print(Dota)

#num_items = 100
#num_games = 891404

num_items = 757
num_games = 10442

#num_items = 4381
#num_games = 61657

#num_items = 1682
#num_games = 139982

#num_items = 2287
#num_games = 28582

print("num_items = ", num_items)

#Dota = Dota[num_items+1:,0:]
#Dota = Dota.astype('int')
Dota = Dota[1:,:]

print("Dota = ", Dota, "size = ", np.shape(Dota)[0])
#print(int(Dota[0][2]), int(Dota[0][3]))
#print(int(Dota[0][1]), int(Dota[0][2]))
print(Dota)

'''num_items = 757 #Dota[0,0]
#num_items = 100
print("num_items = ", num_items)
data_size = Dota[0,0]
print("data_size = ", data_size)

Dota = Dota[1:,:]
print("Dota = ", Dota, "size = ", np.shape(Dota)[0])'''

#seed_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]
for seed in range(1):
 
    #train_data1, train_data2, train_val_data1, train_val_data2, val_data1, val_data2, test_data1, test_data2, prob_data, prob_val = Real_Data.Real_Dataset(Dota, int(num_items), seed)

    #data1 = np.append(train_data1,test_data1,axis = 0)
    #data2 = np.append(train_data2,test_data2,axis = 0)
    #data_final = create_data.create_data(data1, data2, num_items, seed)

    #train_data, test_data = train_test_split(Dota, test_size=0.30, random_state=seed)

    '''with open('WoL_Gnnrank_train0_preprocessing.pkl', 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump(train_data, f)'''

    '''# Getting back the objects:
    with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
        obj0, obj1, obj2 = pickle.load(f)'''

    #np.savetxt("DOTA2_Gnnrank_train0_preprocessing.txt", train_data)
    #np.savetxt("DOTA2_Gnnrank_test0_preprocessing.txt", test_data)
    
    #train_val_data, val_data = train_test_split(train_data, test_size=0.20, random_state=seed)

    #train_data = np.loadtxt("movie_Gnnrank_train0_preprocessing.txt", dtype = 'float')
    #data_final1 = create_data3.create_data(train_data, num_items, 'train', seed)
    test_data = np.loadtxt("DOTA2_Gnnrank_test0_preprocessing.txt", dtype = 'float')
    data_final2 = create_data3.create_data(test_data, num_items, 'test', seed)

    #data_final1 = create_data2.create_data(train_data1, train_data2, num_items, 'train', seed)
    #data_final2 = create_data2.create_data(test_data1, test_data2, num_items, 'test', seed)



