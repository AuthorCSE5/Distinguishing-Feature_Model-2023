import numpy as np
import random
import math
from scipy.special import comb
import time


def create_data(train_val_data, val_data, test_data, num_items, seed, data_name):
    
    train_set = np.copy(train_val_data)
    val_set = np.copy(val_data)
    test_set = np.copy(test_data)
    
    prob_test = np.copy(test_data)

    #np.savetxt("BC_prob_test_"+data_name+"_Real_Data_"+str(seed)+".txt", prob_test)   
    
    print("Creating Training Data...")

    train_size = train_set.shape[0]
    
    data = np.zeros((train_size, 2), dtype = '<U32')
        
    for i in range(train_size):

        data[i,0] = str(train_set[i,0])+':'+str(train_set[i,1])
 
        data[i,1] = str(train_set[i,2])+':'+str(train_set[i,3])

    print("train data = ", data)
        

    print("Creating Validation Data...")

    val_size = val_set.shape[0]
    
    data_val = np.zeros((val_size, 2), dtype = '<U32')
        
    for i in range(val_size):

        data_val[i,0] = str(val_set[i,0])+':'+str(val_set[i,1])
 
        data_val[i,1] = str(val_set[i,2])+':'+str(val_set[i,3])
    
    print("Creating Test Data...")

    test_size = test_set.shape[0]
    
    data_test = np.zeros((test_size, 2), dtype = '<U32')
        
    for i in range(test_size):

        data_test[i,0] = str(test_set[i,0])+':'+str(test_set[i,1])
 
        data_test[i,1] = str(test_set[i,2])+':'+str(test_set[i,3])
        
    
    complete_data = np.append(data, data_val, axis = 0)
    complete_data = np.append(complete_data, data_test, axis = 0)

    train_val_test_mask = np.zeros((train_size + val_size + test_size , 1), dtype='<U32')

    for i in range(train_size):
        train_val_test_mask[i] = 'FOR_TRAINING'

    for i in range(val_size):
        train_val_test_mask[i+train_size] = 'FOR_VALIDATION'

    for i in range(test_size):
        train_val_test_mask[i+train_size+val_size] = 'FOR_TESTING'

   
    player_names = np.zeros((num_items+2, 2), dtype='<U32')
    player_names[0][0] = "numPlayers: "
    player_names[0][1] = str(num_items)
    
    for i in range(num_items):

        player_names[i+1][0] = 'Player'+str(i)+': '
        player_names[i+1][1] = str(i)
        
    num_games = train_size + val_size + test_size
    print(num_games)
    
    player_names[i+2][0] = "numGames: "
    player_names[i+2][1] = str(num_games)
    


    complete_data2 = np.append(train_val_test_mask, complete_data, axis = 1)

    
    print(complete_data2)


    with open('BC_'+data_name+'data_train_Real_data_'+str(seed)+'.txt','w') as f:
        np.savetxt(f, player_names, fmt='%s', delimiter=" ")
        np.savetxt(f, complete_data2, fmt='%s', delimiter=" ")
        #np.savetxt(f, c.reshape(1, -1), fmt='%1.3f', delimiter=",")
    
    
    return complete_data

