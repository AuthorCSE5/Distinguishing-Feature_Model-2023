import numpy as np
from numpy import random
import math
from scipy import stats
import matplotlib.pyplot as plt
from scipy.special import comb

import GNNRank_Accuracy

num_items = 100
total_pairs = int(comb(num_items,2))
m2 = int(num_items * math.log2(num_items))
r = int(total_pairs/m2)
m_vals = []
for i in range(1,r):
    m3 = i*m2
    m_vals.append(m3)

no_of_pairs = m_vals

accuracy_total = np.zeros(((10,r-1)))
ktc_total = np.zeros(((10,r-1)))
rmse_total = np.zeros(((10,r-1)))

count_iter = 0




for seed in range(10):


    ##DFData

    embeddings = np.loadtxt("embeddings"+str(seed)+".txt", dtype = float)

    U = np.copy(embeddings)

    prob_data = np.zeros((num_items,(num_items+1)))

    for i in range(num_items-1):
        for j in range(i+1, num_items):

            diff = np.abs(U[:,i] - U[:,j])
            diff2 = np.amax(diff)
            ind = np.argmax(diff)
            diff3 = (U[ind,i] - U[ind,j])
            prob = 1/(1+np.exp(-diff3))
            #print(prob)
            
            prob_data[i][j] = prob
            prob_data[j][i] = 1 - prob
               

            if prob > 0.5:
                   
                prob_data[i][num_items] = prob_data[i][num_items] + 1
                   
            elif prob < 0.5:
               
               prob_data[j][num_items] = prob_data[j][num_items] + 1
               

    #print("prob_data = ", prob_data)
    #np.savetxt("prob_data_test.txt", prob_data)
    
    
    score_DF = np.zeros(num_items)
    

    for i in range(num_items):
        score_DF[i] = prob_data[i][num_items]

    score_comp = score_DF
        
    '''

    ##BTLData

    embeddings = np.loadtxt("embeddings_BTL"+str(seed)+".txt", dtype = float)

    U = np.copy(embeddings)

    prob_data = np.zeros((num_items,(num_items+1)))

    for i in range(num_items-1):
        for j in range(i+1, num_items):

            
            prob = U[i]/(U[i] + U[j])
            
            
            prob_data[i][j] = prob
            prob_data[j][i] = 1 - prob
               


    score_comp = U

    

    ##SFData

    embeddings = np.loadtxt("embeddings_SF"+str(seed)+".txt", dtype = float)

    SF_weights = np.loadtxt("SFmodel_weights"+str(seed)+".txt", dtype = float)

    U = np.copy(embeddings)

    w_SF = np.copy(SF_weights)

    prob_data = np.zeros((num_items,(num_items+1)))

    for i in range(num_items-1):
        for j in range(i+1, num_items):

            diff = np.abs(U[:,i] - U[:,j])
            diff2 = np.amax(diff)
            ind = np.argmax(diff)
            diff3 = (U[ind,i] - U[ind,j])*w_SF[ind,0]
            prob = 1/(1+np.exp(-diff3))
            #print(prob)
            
            prob_data[i][j] = prob
            prob_data[j][i] = 1 - prob
               

            if prob > 0.5:
                   
                prob_data[i][num_items] = prob_data[i][num_items] + 1
                   
            elif prob < 0.5:
               
               prob_data[j][num_items] = prob_data[j][num_items] + 1
               

    
    
    score_SF = np.zeros(num_items)
    

    for i in range(num_items):
        score_SF[i] = prob_data[i][num_items]

    score_comp = score_SF
        
    '''
    

    count_m = 0

    #m_vals = [664]
    
    for m in m_vals:

        print(m)

        score = np.loadtxt("GNNRank_Datasets_finer/Synthetic/SF/"+str(m)+"/"+str(seed)+"/Score_Dist.txt", dtype = float)
        #score = np.loadtxt("GNNRank_Datasets_finer/Synthetic/"+str(m)+"/"+"64_0.05/Score_innerproduct.txt", dtype = float)
        #score = np.loadtxt("GNNRank_Datasets_finer/Synthetic/BTL/"+str(m)+"/"+str(seed)+"/Score_proximalbaseline.txt", dtype = float)

        #testdata1 = np.loadtxt("GNNRank_Datasets_finer/Synthetic/SF/"+str(m)+"_"+str(seed)+"_testdata1_SF.txt", dtype = float)
        testdata1 = np.loadtxt("GNNRank_Datasets_finer/Synthetic/"+str(m)+"_"+str(seed)+"_testdata1.txt", dtype = float)

        #testdata2 = np.loadtxt("GNNRank_Datasets_finer/Synthetic/SF/"+str(m)+"_"+str(seed)+"_testdata2_SF.txt", dtype = float)
        testdata2 = np.loadtxt("GNNRank_Datasets_finer/Synthetic/"+str(m)+"_"+str(seed)+"_testdata2.txt", dtype = float)

        num_pairs = testdata1.shape[0]

        test_data1 = testdata1[:,:-1]
        test_data2 = testdata2[:,:]

        for i in range(num_pairs):  #test pairs
        
            ind1 = np.argmax(test_data1[i,:])
            ind2 = np.argmax(test_data2[i,:])

            
            #print("prob_data = ",prob_data[ind1][ind2])

        test_acc, ktc, rmse = GNNRank_Accuracy.get_accuracy_synthetic(testdata1, testdata2, prob_data, score_comp, score, num_items)
        

        accuracy_total[count_iter][count_m] = test_acc
        ktc_total[count_iter][count_m] = ktc
        rmse_total[count_iter][count_m] = rmse

        count_m = count_m + 1
    
    count_iter = count_iter + 1


np.savetxt("accuracy_GNN_dist_SFData.txt", accuracy_total)
np.savetxt("ktc_GNN_dist_SFData.txt", ktc_total)
np.savetxt("rmse_GNN_dist_SFData.txt", rmse_total)

