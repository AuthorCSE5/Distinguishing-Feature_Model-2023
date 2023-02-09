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
#import Blade_Chest
import majority_vote3
import LRPR
import GNNRank_Accuracy
import create_data_GNNRank


#Dota = np.array(pd.read_csv('DOTA.csv')) #num_items = 757
#Dota = np.array(pd.read_csv('jester.csv')) #num_items = 100
#Dota = np.array(pd.read_csv('movielens_100k.csv')) #num_items = 1682
#Dota = np.array(pd.read_csv('HotS_Starcraft.csv')) #num_items = 2287
Dota = np.array(pd.read_csv('WoL_Starcraft.csv'))

num_items = 4381


print("num_items = ", num_items)
data_size = Dota[0,0]
print("data_size = ", data_size)



#Dota = Dota[1:,:]  ##used for jester and DoTA
Dota = Dota[num_items+1:,0:]  ##used for HotS and WoL
#Dota = Dota[num_items+1:,1:]  ##used for movielens
Dota = Dota.astype('int')
print("Dota = ", Dota, "size = ", np.shape(Dota)[0])

RC_accuracy_total = np.zeros((10,1))
LRPR_accuracy_total = np.zeros((10,1))
SF_accuracy_total = np.zeros((10,1))
DF_accuracy_total = np.zeros((10,1))
BC_accuracy_total = np.zeros((10,1))
MV_test_ll_total  = np.zeros((10,1))
MV_accuracy_total = np.zeros((10,1))

RC_rmse_total = np.zeros((10,1))
LRPR_rmse_total = np.zeros((10,1))
SF_rmse_total = np.zeros((10,1))

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

def initialize_params_uniform(num_items, dim_mv):
    
    params = np.random.uniform(low = -1, high = 1, size = (dim_mv,num_items))

    return params



#val_loop = [0, 1, 2, 3 ,4 ,5 ,6 ,7 ,8 ,9]

for seed in range(10):

    params = initialize_params_normal(num_items, dim, 1/math.sqrt(0.01), seed)

    train_data1, train_data2, train_val_data1, train_val_data2, val_data1, val_data2, test_data1, test_data2, prob_data, prob_val = Real_Data.Real_Dataset(Dota, int(num_items), seed)

    #RC_accuracy = RC.RankCentrality(train_data1, train_data2, train_val_data1, train_val_data2, val_data1, val_data2, test_data1, test_data2, prob_data, prob_val, num_items)
    RC_accuracy, RC_rmse  = RC_real.RankCentrality(train_data1, train_data2, test_data1, test_data2, num_items, prob_data)
    print("RC_accuracy = ", RC_accuracy)

    SF_accuracy, SF_rmse = SFmodel_MLE_RealData.SF_MLE_RealData(train_data1, train_data2, train_val_data1, train_val_data2, val_data1, val_data2, test_data1, test_data2, prob_data, prob_val, num_items)
    print("SF_accuracy = ", SF_accuracy)

    DF_accuracy, rmse = DFlearn_Realdata2.Ranking(train_data1, train_data2, test_data1, test_data2, num_items, dim2, c, prob_data)
    print("DF_accuracy = ", DF_accuracy)

    
    train_data, test_data = train_test_split(Dota, test_size=0.30, random_state=seed)
    train_val_data, val_data = train_test_split(train_data, test_size=0.20, random_state=seed)

    ##test_ll, test_acc, rmse = majority_vote3.MV_MLE_Realdata(train_data1, train_data2, test_data1, test_data2, num_items, prob_data, params, dim_mv, seed)
    test_ll, test_acc, rmse = majority_vote3.MV_MLE_Realdata(train_data, val_data, test_data, num_items, params, dim_mv, seed)
    print("rmse = ", rmse)
    print("test_ll = ", test_ll)
    print("MV_accuracy = ", test_acc) 

    LRPR_accuracy, LRPR_rmse = LRPR.LRPR_RealData(train_data1, train_data2, test_data1, test_data2, num_items, prob_data)
    print("LRPR_accuracy = ", LRPR_accuracy)

    #GNN_train = create_data3.create_data(train_data, num_items, 'train', seed)
    #GNN_test = create_data3.create_data(test_data, num_items, 'test', seed)

    #np.savetxt("WoL_modifiedtrain0.txt", GNN_train)
    #np.savetxt("WoL_modifiedtest0.txt", GNN_test)
     
    score = np.loadtxt("GNNRank_Datasets_finer/WoL/Score_proximaldist.txt", dtype = float)
    
    testdata = np.loadtxt("GNNRank_Datasets_finer/WoL_modifiedtest0.txt", dtype = float)
        
    GNN_Accuracy = GNNRank_Accuracy.get_accuracy(testdata, score, num_items)
    
    RC_accuracy_total[seed][0] = RC_accuracy
    LRPR_accuracy_total[seed][0] = LRPR_accuracy

    RC_rmse_total[seed][0] = RC_rmse
    LRPR_rmse_total[seed][0] = LRPR_rmse
    
    SF_accuracy_total[seed][0] = SF_accuracy
    SF_rmse_total[seed][0] = SF_rmse
    DF_accuracy_total[seed][0] = DF_accuracy
    BC_accuracy_total[seed][0] = BC_accuracy
    MV_test_ll_total[seed][0] = test_ll
    MV_accuracy_total[seed][0] = test_acc






np.savetxt("accuracy_RC_WoL.txt", RC_accuracy_total)
np.savetxt("accuracy_LRPR_WoL.txt", LRPR_accuracy_total)
np.savetxt("accuracy_DF_WoL.txt", DF_accuracy_total)
np.savetxt("accuracy_SF_WoL.txt", SF_accuracy_total)
np.savetxt("test_ll_MV_WoL.txt", MV_test_ll_total)
np.savetxt("accuracy_MV_WoL.txt", MV_accuracy_total)

np.savetxt("RMSE_RC_WoL.txt", RC_rmse_total)
np.savetxt("RMSE_LRPR_WoL.txt", LRPR_rmse_total)

np.savetxt("RMSE_SF_WoL.txt", SF_accuracy_total)
np.savetxt("RMSE_DF_WoL.txt", SF_rmse_total)

np.savetxt("accuracy_BC_WoL.txt", BC_accuracy_total)
np.savetxt("RMSE_BC_WoL.txt", BC_rmse_total)





