import numpy as np
import random
import math
from scipy.special import comb
import time


def create_data(train_set1, train_set2, num_items, flag, t):

    print("Creating Data...")

    data_set1 = np.copy(train_set1)
    data_set2 = np.copy(train_set2)
        
    total_pairs = int(comb(num_items,2))

    s = data_set1.shape[0]
    
    ts1 = data_set1[:,:-1]
    ts2 = data_set2[:,:]

    
    ts3 = np.append(ts1,ts2,axis = 1)

    data_set3 = ts3

    ts3_original = ts3
    ts1_original = ts1
    ts2_original = ts2
    
    ts3 = np.unique(ts3,axis = 0)
    s2 = ts3.shape[0]

    num_pairs = int(s2)

    ts31 = ts3[:,0:num_items]
    ts32 = ts3[:,num_items:2*num_items]

    s3 = ts31.shape[0]
        
    count = 0
    data = np.zeros((s2, 4))
    count_data = s
    

    print("s2 = ", s2)
    
    for i in range(s2):

        
        ind1 = np.argmax(ts31[i,:])
        ind2 = np.argmax(ts32[i,:])

        data[i,0] = ind1
        data[i,1] = ind2


        argw1 = np.argwhere(ts1_original == ts1_original.max())
        argw2 = np.argwhere(ts2_original == ts2_original.max())
        
        argw = np.append(argw1,argw2,axis = 1)

        mask1 = (argw[:,1] == ind1)
        mask2 = (argw[:,3] == ind2)
       
        mask_where = np.where(mask1 & mask2)

        
        final_data = data_set1[mask_where]
        
        final_data_sign = np.copy(final_data[:,num_items])

        
        
        data[i,2] = np.sum(final_data_sign == 1)
        data[i,3] = np.sum(final_data_sign == -1)

        
    np.savetxt("BTLData_"+flag+"_"+str(t)+".txt", data)
        
    return data, num_pairs
