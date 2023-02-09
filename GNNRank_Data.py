import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from scipy.special import comb
import pandas as pd
from sklearn.model_selection import train_test_split

import Real_Data
import create_data3


#Dota = np.array(pd.read_csv('DOTA.csv')) #also used for jester.csv
#Dota = np.array(pd.read_csv('jester.csv'))
#Dota = np.array(pd.read_csv('WoL_Starcraft.csv'))
#Dota = np.array(pd.read_csv('movielens_100k.csv'))
Dota = np.array(pd.read_csv('HotS_Starcraft.csv'))
print(Dota)

#num_items = 100
#num_games = 891404
#K = int(math.log2(num_items)) + 1 #ceiling
#K = 7

'''num_items = 757
num_games = 10442
K = 10'''

'''num_items = 4381
num_games = 61657
K = 13'''

'''num_items = 1682
num_games = 139982
K = 11'''

num_items = 2287
num_games = 28582
K = 12     #K = int(math.log2(num_items)) + 1 #ceiling

print("num_items = ", num_items)

Dota = Dota[num_items+1:,0:]  ##for HotS and WoL
#Dota = Dota[num_items+1:,1:]  ##for movielens
Dota = Dota.astype('int')
#Dota = Dota[1:,:]  ##for jester and Dota

print("Dota = ", Dota, "size = ", np.shape(Dota)[0])

for seed in range(10):
 
    #train_data1, train_data2, train_val_data1, train_val_data2, val_data1, val_data2, test_data1, test_data2, prob_data, prob_val = Real_Data.Real_Dataset(Dota, int(num_items), seed)

    #data1 = np.append(train_data1,test_data1,axis = 0)
    #data2 = np.append(train_data2,test_data2,axis = 0)
    #data_final = create_data.create_data(data1, data2, num_items, seed)

    train_data, test_data = train_test_split(Dota, test_size=0.30, random_state=seed)
    train_val_data, val_data = train_test_split(train_data, test_size=0.20, random_state=seed)

    np.savetxt("GNNRank_Datasets_finer/HotS_Gnnrank_train"+str(seed)+"_preprocessing.txt", train_data)
    np.savetxt("GNNRank_Datasets_finer/HotS_Gnnrank_test"+str(seed)+"_preprocessing.txt", test_data)
    
    
    #train_data = np.loadtxt("GNNRank_Datasets_finer/DOTA_Gnnrank_train"+str(seed)+"_preprocessing.txt", dtype = 'float')
    data_final1, train_pairs = create_data3.create_data(train_data, num_items, 'train', seed)
    #test_data = np.loadtxt("GNNRank_Datasets_finer/HotS_Gnnrank_test"+str(seed)+"_preprocessing.txt", dtype = 'float')
    data_final2, test_pairs = create_data3.create_data(test_data, num_items, 'test', seed)

    

