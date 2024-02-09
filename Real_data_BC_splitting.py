import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt
import pickle
from scipy.special import comb
import pandas as pd
from sklearn.model_selection import train_test_split

import Real_Data
import DFlearn_Realdata2
import RC_real
import SFmodel_MLE_RealData
import Blade_Chest
import majority_vote3
import LRPR
import GNNRank_Accuracy
import create_data3
import BC_Accuracy

import Real_Data_Sushi
import DFlearn_sushi

import Real_data_splitting
import BC_Accuracy_new
from data_convert import *


Dota = np.array(pd.read_csv('SF07HumanRead.csv'))

data_name = 'SF07HumanRead' #'WoL_Starcraft' #'tennis_new'
num_items = 18
#num_items = 10
#num_items = 1682
#num_items = 2287
#num_items = 35

#num_items = 757 #Dota[0,0]

#num_items = 4381


print("num_items = ", num_items)
data_size = Dota[num_items,1]
print("data_size = ", data_size)

dim2 = 10
c = 0.000009 #l2 regularization parameter


print(Dota)
#print(np.shape(Dota))


#Dota = Dota[1:,:]
Dota = Dota[num_items+1:,0:]
print("Dota = ", Dota, "size = ", np.shape(Dota)[0])
Dota = Dota.astype('int')
print("Dota = ", Dota, "size = ", np.shape(Dota)[0])

'''
Dota = np.array(pd.read_csv('tennis_new.csv'))
data_name = 'tennis_new' #'WoL_Starcraft' #'tennis_new'

num_items = 742
data_size = Dota[num_items,1]

print("Dota = ", Dota)
Dota2 = Dota[num_items+1:,0:4]
Dota2 = Dota2.astype('int')
print("Dota = ", Dota2)
dt_size = Dota2.shape[0]
print(dt_size)
print(type(Dota2))

Dota = convert(Dota2, num_items, data_size, dt_size)
print("Dota = ", Dota)
'''
BC_accuracy_total = np.zeros((10,1))
BC_rmse_total = np.zeros((10,1))
MV_rmse_total  = np.zeros((10,1))
MV_accuracy_total = np.zeros((10,1))

lb = 0.001
choice = 'inner'
dim = 3
dim_mv = 3
dim = dim_mv

def initialize_params_normal(num_items, dim_mv, std_deviation,seed):

    np.random.seed(seed)
        
    params = np.random.normal(0, scale = std_deviation, size = (dim_mv,num_items))

    params = params/np.linalg.norm(params,2,axis = 0)
       
    return params

def initialize_params_uniform(num_items, dim_mv, seed):

    np.random.seed(seed)
    
    params = np.random.uniform(low = -1, high = 1, size = (dim_mv,num_items))

    #params = params/np.linalg.norm(params,2,axis = 0)

    return params



val_loop = [7, 8, 9] # ,4 ,5 ,6 ,7 ,8 ,9]

for seed in range(10):#val_loop:

    #params = initialize_params_normal(num_items, dim_mv, 1/math.sqrt(0.01), seed)
    params = initialize_params_uniform(num_items, dim_mv, seed)
    
    '''
    train_data, test_data = train_test_split(Dota, test_size=0.30, random_state=seed)
    train_val_data, val_data = train_test_split(train_data, test_size=0.20, random_state=seed)

    print("train_val_data = ", train_val_data)
    print("val_data = ", val_data)
    print("test data = ", test_data)
    
    
    Real_data_splitting.create_data(train_val_data, val_data, test_data, num_items, seed, data_name)
    '''
    '''
    test_acc_bc, pred_accuracy_bc, rmse_bc = BC_Accuracy_new.BC(data_name, num_items, seed)
    print("BC_accuracy = ", pred_accuracy_bc)
    print("BC_rmse = ", rmse_bc)
    '''
    
    train_data, test_data = train_test_split(Dota, test_size=0.30, random_state=seed)
    train_val_data, val_data = train_test_split(train_data, test_size=0.20, random_state=seed)
    
    ##test_ll, test_acc, rmse = majority_vote3.MV_MLE_Realdata(train_data1, train_data2, test_data1, test_data2, num_items, prob_data, params, dim_mv, seed)
    test_ll_mv, test_acc_mv, rmse_mv = majority_vote3.MV_MLE_Realdata(train_data, val_data, test_data, num_items, params, dim_mv, seed, data_name)
    #test_ll, test_acc, rmse = majority_vote3.MV_MLE_Realdata(num_items, params, dim_mv, seed)
    print("rmse = ", rmse_mv)
    print("test_ll = ", test_ll_mv)
    print("MV_accuracy = ", test_acc_mv)

    MV_rmse_total[seed][0] = rmse_mv
    MV_accuracy_total[seed][0] = test_acc_mv
    
    
    #BC_accuracy_total[seed][0] = pred_accuracy_bc
    #BC_rmse_total[seed][0] = rmse_bc

    #train_data, test_data = train_test_split(Dota, test_size=0.30, random_state=seed)

    #np.savetxt("test_data_gnnrank_sushi3A.txt", test_data)

    #test_data = np.loadtxt("test_data_gnnrank_sushi3B.txt", dtype = float)
    
        
    np.savetxt("accuracy_MV_sf07_momentum_0.00005.txt", MV_accuracy_total)
    np.savetxt("rmse_MV_sf07_momentum_0.00005.txt", MV_rmse_total)

    #np.savetxt("rmse_MV_DW.txt", MV_rmse_total)
    #np.savetxt("accuracy_MV_DW.txt", MV_accuracy_total)




