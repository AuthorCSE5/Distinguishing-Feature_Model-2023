import numpy as np
import random
import math
from scipy.special import comb
import time


def create_data(train_set, num_items, flag, t):

    print("Creating Data...")

    data_set1 = np.copy(train_set)

    data_set1 = np.asarray(data_set1, dtype = 'int')

    print("data_set1 = ", data_set1)

    data_set1_temp = np.copy(data_set1)

    #data_set1_temp = np.sort(data_set1_temp, axis = 0)
    #print("data_set1_temp = ", data_set1_temp)
        
    total_pairs = int(comb(num_items,2))

    s = data_set1.shape[0]
    
    ts1 = data_set1[:,[0,1]]
    
    ts1 = np.unique(ts1,axis = 0)

    #ts1,idx = np.unique(ts1,axis = 0, return_index=True)
    
    #print("ts1_unique = ", ts1)
    #print(idx)

    print("ts1 = ", ts1)
    s2 = ts1.shape[0]

    num_pairs = int(s2)


    count = 0
    data = np.zeros((s2, 4))
    count_data = s
    
    print("data = ", data.shape)
    print("s2 = ", s2)

    

    '''for i in range(s):

        ind1 = np.argmax(ts1[i,:])
        ind2 = np.argmax(ts2[i,:])

        test_data[i,0] = ind1
        test_data[i,1] = ind2

        if data_set1[i,num_items] == 1:
            test_data[i,2] = 1
        elif data_set1[i,num_items] == -1:
            test_data[i,3] = 1'''

    #For Gnnrank

    players = []

    
    
    for i in range(s2):

        #start = time.process_time()

        #print("i = ", i)
        
        test_data = np.copy(data_set1)
        
        data[i,0] = ts1[i,0]
        data[i,1] = ts1[i,1]

        #print("test_data = ", test_data)

        #np.savetxt("test_data.txt", test_data)

        test_data1 = np.zeros((s, 4))
        test_data1[:,0] = np.copy(data[i,0])
        test_data1[:,1] = np.copy(data[i,1])

        #print("test_data1 = ", test_data1)

        test_data_final = test_data - test_data1

        #print("test_data_final = ",test_data_final.shape[0])

        #np.savetxt("test_data_final.txt", test_data_final)

        #print("test_data_final_neg = ", test_data_final[test_data_final[:,1] < 0])
        
        '''test_data_final = np.delete(test_data_final, test_data_final[:,1] < 0, axis = 0)
        print("test_data_final1 = ",test_data_final.shape[0])

        test_data_final = np.delete(test_data_final, test_data_final[:,0] < 0, axis = 0)
        print("test_data_final2 = ",test_data_final.shape[0])

        test_data_final = np.delete(test_data_final, test_data_final[:,0] == 1, axis = 0)
        print("test_data_final3 = ",test_data_final.shape[0])

        print("test_data_final_neg2 = ", test_data_final[test_data_final[:,1] < 0])'''

        test_data_final = np.delete(test_data_final, test_data_final[:,1] != 0, axis = 0)
        #print("test_data_final1 = ",test_data_final.shape[0])

        test_data_final = np.delete(test_data_final, test_data_final[:,0] != 0, axis = 0)

        #print("test_data_final = ",test_data_final.shape[0])
        #print("test_data_final = ",test_data_final)

        na_col = test_data_final[:,2]
        nb_col = test_data_final[:,3]

        data[i,2] = np.sum(na_col == 1)
        data[i,3] = np.sum(nb_col == 1)

        '''
        #print("test_data_final1 = ",test_data_final.shape[0])

        #print("test_data_final_delete = ",test_data_final.shape[0])
        

        #mask1 = (test_data_final[:,0] < 0)
        #mask2 = (test_data_final[:,1] < 0)
       
        #mask_where = np.where(mask1 | mask2)

        
        #test_data_final_mask = test_data_final#[mask_where]

        #test_data_final[:,[2,3]] = test_data[:,[2,3]]

        test_data_final1 = test_data_final[:,0:3]
        test_data_final2 = test_data_final[:,[0,1,3]]

        #print("test_data_final4 = ",test_data_final.shape[0])

        #print("test_data_final_mask = ",test_data_final)

        

        sum_row1 = np.sum(test_data_final1, axis = 1)
        sum_row2 = np.sum(test_data_final2, axis = 1)

        #print("test_data_final5 = ",test_data_final.shape[0])
        
        print("sum_row1 = ",sum_row1)
        print("sum_row2 = ",sum_row2)

        na_col = test_data_final[:,2]
        nb_col = test_data_final[:,3]

        print("test_data_final6 = ",test_data_final.shape[0])

        print("na_col = ",na_col)
        print("nb_col = ",nb_col)

        na_count_arr = 0.5*(sum_row1 + na_col)
        nb_count_arr = 0.5*(sum_row2 + nb_col)

        print("test_data_final7 = ",test_data_final.shape[0])

        print("na_count_arr = ",na_count_arr)
        print("nb_count_arr = ",nb_count_arr)

        

        print("np.argwhere(na_count_arr == 1) = ",np.argwhere(na_count_arr == 1))

        indices = np.argwhere(na_count_arr == 1)

        print("test_data_final[indices] = ",test_data_final[indices])

        #test_data_final = np.append(test_data_final, na_count_arr, axis = 1)
        #test_data_final = np.append(test_data_final, nb_count_arr, axis = 1)

        print("test_data_final_appended = ", test_data_final)

        

        if data[i,1] == 12:

            np.savetxt("na_count_arr.txt", na_count_arr)
            np.savetxt("nb_count_arr.txt", nb_count_arr)
            np.savetxt("test_data_final.txt", test_data_final)
            print("test_data_final8 = ",test_data_final.shape[0])

        data[i,2] = np.sum(na_count_arr == 1)
        data[i,3] = np.sum(nb_count_arr == 1)

        #players.append(data[i,0])
        #players.append(data[i,1])

        print("data = ", data.shape)'''
        
        #print(data[i,:])

    '''
    
    data2 = np.copy(data)

    data2[:,[0,1]] = data2[:,[1,0]]
    data2[:,[2,3]] = data2[:,[3,2]]

    print(data.shape[0])

    data = np.unique(data, axis = 0)

    print(data.shape[0])

    data = np.append(data, data2, axis = 0)

    print(data.shape[0])

    data = np.unique(data, axis = 0)

    print(data.shape[0])

    data = np.delete(data, [3] , 1)

    
    
    #not for jester (for connected components and adding all the players)

    players_loop = np.zeros((num_items, 2))

    for i in range(num_items-1):

        players_loop[i,0] = i
        players_loop[i,1] = i+1

    players_loop[num_items-1,0] = num_items - 1
    players_loop[num_items-1,1] = 0

    players_loop2 = np.copy(players_loop)

    players_loop2[:,[0,1]] = players_loop2[:,[1,0]]

    players_loop = np.append(players_loop, players_loop2, axis = 0)


    data_orig = np.copy(data)

    data_temp = data_orig[:,0:2]

    print(data.shape[0], data_temp.shape[0])

     
    
    players_loop_temp = np.copy(players_loop)
    
    data_temp = np.append(data_temp, players_loop_temp, axis = 0)

    print(data.shape[0], data_temp.shape[0])

    data_temp = np.unique(data_temp, axis = 0)

    print(data.shape[0], data_temp.shape[0])

    data = np.unique(data, axis = 0)

    print(data.shape[0], data_temp.shape[0])

    players_loop_size = data_temp.shape[0] - data.shape[0]

    print("players_loop_size = ", players_loop_size)

    players_loop_win = np.zeros((players_loop_size,1))

    original_wins_1 = data[:,2:3]

    original_wins = np.copy(original_wins_1)

    original_wins = np.append(original_wins, players_loop_win, axis = 0)

    data = np.append(data_temp, original_wins, axis = 1)

    print(data)

    ## above not for jester  

    data[:,2] = data[:,2] + 0.1     '''           

    print(data)

    
    #np.savetxt("WoL_modified"+flag+str(t)+".txt", data)
        
    np.savetxt("movie_modified"+flag+str(t)+".txt", data)
    
    #np.savetxt("GNNRank_Datasets_finer/HotS_modified"+flag+str(t)+".txt", data)

    #np.savetxt("HotS_GNNRank"+flag+str(t)+".txt", data)

    #np.savetxt("jester_BC_test" + str(num_pairs) + ".txt", data)
        
    return data, num_pairs


def create_data_synthetic(train_set1, train_set2, num_items, seed, m):

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
    
   
    '''
    #For Gnnrank

    players = []

    
    
    for i in range(s2):

        ind1 = np.argmax(ts31[i,:])
        ind2 = np.argmax(ts32[i,:])

        
        data[i,0] = ind1
        data[i,1] = ind2

        test_data1 = np.zeros((s, 4))
        test_data1[:,0] = data[i,0]
        test_data1[:,1] = data[i,1]

        test_data_final = test_data - test_data1

        test_data_final1 = test_data_final[:,0:3]
        test_data_final2 = test_data_final[:,[0,1,3]]

        sum_row1 = np.sum(test_data_final1, axis = 1)
        sum_row2 = np.sum(test_data_final2, axis = 1)

        na_col = test_data_final[:,2]
        nb_col = test_data_final[:,3]

        na_count_arr = 0.5*(sum_row1 + na_col)
        nb_count_arr = 0.5*(sum_row2 + nb_col)

        data[i,2] = np.sum(na_count_arr == 1)
        data[i,3] = np.sum(nb_count_arr == 1)

        players.append(data[i,0])
        players.append(data[i,1])

        #print("data = ", data.shape)
        
        #print(data[i,:])

    '''

    
    
    data2 = np.copy(data)

    data2[:,[0,1]] = data2[:,[1,0]]
    data2[:,[2,3]] = data2[:,[3,2]]

    print(data.shape[0])

    data = np.unique(data, axis = 0)

    print(data.shape[0])

    data = np.append(data, data2, axis = 0)

    print(data.shape[0])

    data = np.unique(data, axis = 0)

    print(data.shape[0])

    data = np.delete(data, [3] , 1)

    
    '''
    #not for jester (for connected components and adding all the players)

    players_loop = np.zeros((num_items, 2))

    for i in range(num_items-1):

        players_loop[i,0] = i
        players_loop[i,1] = i+1

    players_loop[num_items-1,0] = num_items - 1
    players_loop[num_items-1,1] = 0

    players_loop2 = np.copy(players_loop)

    players_loop2[:,[0,1]] = players_loop2[:,[1,0]]

    players_loop = np.append(players_loop, players_loop2, axis = 0)


    data_orig = np.copy(data)

    data_temp = data_orig[:,0:2]

    print(data.shape[0], data_temp.shape[0])

     
    
    players_loop_temp = np.copy(players_loop)
    
    data_temp = np.append(data_temp, players_loop_temp, axis = 0)

    print(data.shape[0], data_temp.shape[0])

    data_temp = np.unique(data_temp, axis = 0)

    print(data.shape[0], data_temp.shape[0])

    data = np.unique(data, axis = 0)

    print(data.shape[0], data_temp.shape[0])

    players_loop_size = data_temp.shape[0] - data.shape[0]

    print("players_loop_size = ", players_loop_size)

    players_loop_win = np.zeros((players_loop_size,1))

    original_wins_1 = data[:,2:3]

    original_wins = np.copy(original_wins_1)

    original_wins = np.append(original_wins, players_loop_win, axis = 0)

    data = np.append(data_temp, original_wins, axis = 1)

    print(data)

    ## above not for jester  '''

    data[:,2] = data[:,2] + 0.1     

    print(data)   
        
    #np.savetxt("GNNRank_Datasets_finer/Synthetic/"+str(m)+"_"+str(seed)+"_SFData.txt", data)
    np.savetxt("GNNRank_Datasets_finer/Synthetic/"+str(m)+"_"+str(seed)+"_DFData.txt", data)
    #np.savetxt("GNNRank_Datasets_finer/jester_modifiedtrain"+str(seed)+".txt", data)
        
    return data, num_pairs
