import numpy as np
import random
import math
from scipy.special import comb
import time


def create_data(embedding_obj, num_items, l, m, seed, data_name):
    
    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)
    #validation_data1 = np.copy(embedding_obj.validation_data1)
    #validation_data2 = np.copy(embedding_obj.validation_data2)
    prob_test = np.copy(embedding_obj.prob_test)

    np.savetxt("BC_prob_test_"+data_name+"Data_"+str(m)+"_"+str(seed)+".txt", prob_test)   

    total_pairs = int(comb(num_items,2))
    

    print("m = ", m)
   
    
    print("Creating Training Data...")

    data_set1 = np.copy(train_set1)
    data_set2 = np.copy(train_set2)
        
    
    train_size = data_set1.shape[0]
    
    ts1 = data_set1[:,:-1]
    ts2 = data_set2[:,:]

    data = np.zeros((train_size, 2), dtype = '<U32')
        
    for i in range(train_size):

        
        ind1 = np.argmax(ts1[i,:])
        ind2 = np.argmax(ts2[i,:])

        data[i,0] = str(ind1)+':'+str(ind2)
        


               
        if data_set1[i,num_items] == 1:       
        
            data[i,1] = str(1)+':'+str(0)
            

        elif data_set1[i,num_items] == -1:       
        
            data[i,1] = str(0)+':'+str(1)
            

    print("Creating Validation Data...")

    data2 = np.copy(data)

    mask = np.random.choice([False, True], len(data2), p=[0.75, 0.25])

    data_val = data2[mask]

    val_size = data_val.shape[0]
    
    print("Creating Test Data...")

    data_set1 = np.copy(test_set1)
    data_set2 = np.copy(test_set2)
        
    
    test_size = data_set1.shape[0]
    
    ts1 = data_set1[:,:-1]
    ts2 = data_set2[:,:]

    data_test = np.zeros((test_size, 2), dtype = '<U32')
        
    for i in range(test_size):

        
        ind1 = np.argmax(ts1[i,:])
        ind2 = np.argmax(ts2[i,:])

        data_test[i,0] = str(ind1)+':'+str(ind2)
        


               
        if data_set1[i,num_items] == 1:       
        
            
            data_test[i,1] = str(1)+':'+str(0)

        elif data_set1[i,num_items] == -1:       
        
            
            data_test[i,1] = str(0)+':'+str(1)

    
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
    
        
    

    with open('BC_'+data_name+'data_train'+str(m)+'_'+str(seed)+'.txt','w') as f:             
        np.savetxt(f, player_names, fmt='%s', delimiter=" ")
        np.savetxt(f, complete_data2, fmt='%s', delimiter=" ")
        
    
    
    return complete_data

