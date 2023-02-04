import pandas as pd
import numpy as np
import os
import sys
import math
from scipy.special import comb
from sklearn.model_selection import train_test_split


def Real_Dataset(Dota,num_items,seed):

    s = np.shape(Dota)[0]
    num_items = int(num_items)

    m = int(comb(num_items,2))#no. of pairs
    l = s #no. of comparisons per pair

    data_size = s

    data1 = np.zeros((data_size, (num_items + 1)))
    data2 = np.zeros((data_size, num_items))

    train_size = int(data_size * 0.7)
    test_size = data_size - train_size

    train_data1 = np.zeros((train_size, (num_items + 1)))
    train_data2 = np.zeros((train_size, num_items))

    test_data1 = np.zeros((test_size, (num_items + 1)))
    test_data2 = np.zeros((test_size, num_items))

    #rank_data1 = np.zeros((int(comb(num_items,2)),num_items))
    #rank_data2 = np.zeros((int(comb(num_items,2)),num_items))

       
    '''w = 0
            
    for i in range(num_items-1):
        for j in range(i+1, num_items):
            rank_data1[w][i] = 1
            rank_data2[w][j] = 1
        
            w = w + 1

    '''


    
    for z in range(data_size):
        i = int(Dota[z][0])
        j = int(Dota[z][1])
            
        
        if (int(Dota[z][2]) > int(Dota[z][3])):
           data1[z][i] = 1
           data2[z][j] = 1
           data1[z][num_items] = 1
           
        elif (int(Dota[z][2]) < int(Dota[z][3])):
           data1[z][i] = 1
           data2[z][j] = 1
           data1[z][num_items] = -1
           


    data = np.append(data1, data2, axis = 1)
    print("data1 = ", data1)

    '''np.random.shuffle(data)

    for i in range(train_size):
        train_data1[i][:] = data[i][0:num_items+1]
        train_data2[i][:] = data[i][num_items+1:(2*(num_items)+1)]

    print("train_data = ", np.shape(train_data1))


    for i in range(train_size, train_size + test_size):
        test_data1[i-train_size][:] = data[i][0:num_items+1]
        test_data2[i-train_size][:] = data[i][num_items+1:(2*(num_items)+1)]

    print("test_data = ", np.shape(test_data1))

    train_data = np.append(train_data1, train_data2, axis = 1)
    test_data = np.append(test_data1, test_data2, axis = 1)'''


    train_data, test_data = train_test_split(data, test_size=0.30, random_state=seed)
    train_val_data, val_data = train_test_split(train_data, test_size=0.20, random_state=seed)

    train_size = np.shape(train_data)[0]
    test_size = np.shape(test_data)[0]
    train_val_size = np.shape(train_val_data)[0]
    val_size = np.shape(val_data)[0]

    train_data1 = np.zeros((train_size, (num_items + 1)))
    train_data2 = np.zeros((train_size, num_items))

    test_data1 = np.zeros((test_size, (num_items + 1)))
    test_data2 = np.zeros((test_size, num_items))

    train_val_data1 = np.zeros((train_val_size, (num_items + 1)))
    train_val_data2 = np.zeros((train_val_size, num_items))

    val_data1 = np.zeros((val_size, (num_items + 1)))
    val_data2 = np.zeros((val_size, num_items))

    #print("train_data = ", train_data)
    #print("test_data = ", test_data)

    prob_data = np.zeros((num_items, num_items))
    prob_val = np.zeros((num_items, num_items))
    count_data = np.zeros((num_items, num_items)) #not regularize for test data, only for training purpose used
    count_val = np.zeros((num_items, num_items))

    for i in range(train_size):
        train_data1[i][:] = train_data[i][0:num_items+1]
        train_data2[i][:] = train_data[i][num_items+1:(2*(num_items)+1)]

    for i in range(train_val_size):
        train_val_data1[i][:] = train_val_data[i][0:num_items+1]
        train_val_data2[i][:] = train_val_data[i][num_items+1:(2*(num_items)+1)]

    for i in range(val_size):
        val_data1[i][:] = val_data[i][0:num_items+1]
        val_data2[i][:] = val_data[i][num_items+1:(2*(num_items)+1)]

    for i in range(test_size):
        test_data1[i][:] = test_data[i][0:num_items+1]
        test_data2[i][:] = test_data[i][num_items+1:(2*(num_items)+1)]
    
    for t in range(test_size):
        ind1 = np.argmax(test_data1[t,:-1])
        ind2 = np.argmax(test_data2[t,:])
                
        #print("t = ", t)
        if test_data1[t][ind1] == 1 and test_data2[t][ind2] == 1:
            if test_data1[t][num_items] == 1:
               count_data[ind1][ind2] += 1
            elif test_data1[t][num_items] == -1:
               count_data[ind2][ind1] += 1

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if count_data[i][j] + count_data[j][i] > 0:
                prob_data[i][j] = count_data[i][j]/(count_data[i][j] + count_data[j][i])
                prob_data[j][i] = 1 - prob_data[i][j]

    print("prob_data = ", prob_data)

    for t in range(val_size):
        ind1 = np.argmax(val_data1[t,:-1])
        ind2 = np.argmax(val_data2[t,:])
                
        #print("t = ", t)
        if test_data1[t][ind1] == 1 and test_data2[t][ind2] == 1:
            if val_data1[t][num_items] == 1:
               count_val[ind1][ind2] += 1
            elif val_data1[t][num_items] == -1:
               count_val[ind2][ind1] += 1

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if count_val[i][j] + count_val[j][i] > 0:
                prob_val[i][j] = count_val[i][j]/(count_val[i][j] + count_val[j][i])
                prob_val[j][i] = 1 - prob_val[i][j]

    
    return train_data1, train_data2, train_val_data1, train_val_data2, val_data1, val_data2, test_data1, test_data2, prob_data, prob_val#, rank_data1, rank_data2

    #return train_data1, train_data2, test_data1, test_data2, prob_data    
                


