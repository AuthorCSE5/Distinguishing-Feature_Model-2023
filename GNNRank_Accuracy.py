import numpy as np
import pandas as pd
import math
from scipy.special import comb
from scipy import stats


'''def get_accuracy(test_set1, test_set2, prob_data, score, num_items):

       
    #prediction accuracy on test data

    pred_accuracy = 0
    prob_GNNRank = np.zeros((num_items, num_items))

    for i in range(0, num_items):
        for j in range(i+1, num_items):
            prob_GNNRank[i][j] = 1/(1+np.exp(-(score[i]-score[j])))
            prob_GNNRank[j][i] = 1 - prob_GNNRank[i][j]

    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    
    s = test_data1.shape[0]
    #print("s = ", s)

    ind_test = np.zeros((s,2))
    

    for t in range(s):

        
        ind1 = np.argmax(test_data1[t,:])
        ind2 = np.argmax(test_data2[t,:])
        ind_test[t,0] = ind1
        ind_test[t,1] = ind2
        

    test_index2 = np.unique(ind_test, axis = 0)
    
    #print("ind_test = ", ind_test.shape[0])
    #print("text_index2 = ", test_index2.shape[0])
    
    s2 = test_index2.shape[0]

    for t in range(s2):

        
        ind1 = int(test_index2[t,0])
        ind2 = int(test_index2[t,1])
        #ind1 = np.argmax(test_data1[t,:])
        #ind2 = np.argmax(test_data2[t,:])

        #print("test_index2[t,0] = ", test_index2[t,0])
        #print("test_index2[t,1] = ", test_index2[t,1])
    
        if ((prob_data[ind1][ind2] - 0.5)*(prob_GNNRank[ind1][ind2] - 0.5)) > 0:
            pred_accuracy = pred_accuracy + 1
        

    #pred_accuracy = pred_accuracy/s

    pred_accuracy = pred_accuracy/s2

    print("pred_accuracy = ", pred_accuracy)
    
    return pred_accuracy'''

def get_accuracy_synthetic(testdata1, testdata2, prob_data, score_comp, score_orig, num_items):


    testset1 = np.copy(testdata1)
    testset2 = np.copy(testdata2)
    prob_data2 = np.copy(prob_data)
    

    score_DFdata = np.copy(score_comp)

    score = np.copy(score_orig)

    #prediction accuracy on test data

    print("score = ", score)

    #score = score/np.sum(score)

    print("score = ", score)

    #score = np.flip(score)

    pred_accuracy = 0
    prob_GNNRank = np.zeros((num_items, num_items))

    num_pairs = testset1.shape[0]

    for i in range(0, num_items):
        for j in range(i+1, num_items):
            prob_GNNRank[i][j] = 1/(1+np.exp(-(score[i]-score[j]))) 
            prob_GNNRank[j][i] = 1 - prob_GNNRank[i][j]

    accuracy = 0
    rmse = 0
    upsets = 0
    pred_accuracy = 0


    

    prob_GNN = np.zeros(num_pairs)
    prob_test = np.zeros(num_pairs)
    
    total_pairs = int(comb(num_items,2))

    test_data1 = testset1[:,:-1]
    test_data2 = testset2[:,:]

    num_pairs = test_data1.shape[0]
    
    #prob_estimated = np.zeros(num_pairs)
    
    for i in range(num_pairs):  #test pairs
        
        ind1 = np.argmax(test_data1[i,:])
        ind2 = np.argmax(test_data2[i,:])

        #print("prob_GNNRank = ",prob_GNNRank[ind1][ind2])
        #print("prob_data2 = ",prob_data2[ind1][ind2])

        if (prob_GNNRank[ind1][ind2] - 0.5) * (prob_data2[ind1][ind2] - 0.5) > 0:

            pred_accuracy += 1 
             
        

        rmse = rmse + (prob_GNNRank[ind1][ind2] - prob_data2[ind1][ind2])**2

        

    acc1 = pred_accuracy/num_pairs

    rmse = math.sqrt(rmse/num_pairs)

    print("rmse = ", rmse)

    print("accuracy = ", acc1)

    #np.savetxt("prob_GNNRank_DFdata.txt", prob_GNNRank)
    #np.savetxt("prob_test_DFdata.txt", prob_data2)   

    ktc2, p_val = stats.kendalltau(score_DFdata,score)
    print("ktc2 = ",ktc2)
    
    return acc1, ktc2, rmse


    
def get_accuracy(data, score, num_items):

   
    #prediction accuracy on test data

    print("score = ", score)
    #print(sum(score>0.5),len(score))
    #score = score/np.sum(score)

    print("score = ", score)

    pred_accuracy = 0
    prob_GNNRank = np.zeros((num_items, num_items))
    prob_data = np.zeros((num_items, num_items))

    num_pairs = data.shape[0]

    for i in range(0, num_items):
        for j in range(i+1, num_items):
            prob_GNNRank[i][j] = 1/(1+np.exp(-(score[i]-score[j]))) #score[i]/(score[i] + score[j])
            prob_GNNRank[j][i] = 1 - prob_GNNRank[i][j]

            #print("Flag 1",prob_GNNRank[i][j],prob_GNNRank[j][i])

    

    accuracy = 0
    rmse = 0
    upsets = 0
    pred_accuracy = 0
    
    na_total = 0
    nb_total = 0

    prob_GNN = np.zeros(num_pairs)
    prob_test = np.zeros(num_pairs)      
    
    for i in range(num_pairs):

        a = data[i][0]
        b = data[i][1]
        na = data[i][2]
        nb = data[i][3]
        p_a = 0
        p_b = 0
    
        a = int(float(a))
        b = int(float(b))


        prob_data[a][b] = (int(float(na)))/(int(float(na)) + int(float(nb)))
        prob_data[b][a] = 1 - prob_data[a][b]

        ##Prediction Accuracy

        if (prob_GNNRank[a][b] - 0.5) * (prob_data[a][b] - 0.5) > 0:

            pred_accuracy = pred_accuracy + (int(float(na)) + int(float(nb)))


        rmse = rmse + ((prob_GNNRank[a][b] - prob_data[a][b])**2)*(int(float(na)) + int(float(nb)))


        
        
        if prob_GNNRank[a][b] > 0.5:
            p_a += 1
        elif prob_GNNRank[a][b] < 0.5:
            p_b += 1

        #prob_GNN[i] = prob_GNNRank[a][b]
        #prob_test[i] = prob_data[a][b]


            

        if (score[a] - score[b]) * (prob_data[a][b] - 0.5) < 0:
            upsets += (int(float(na)) + int(float(nb)))


            

        #print("na = ", na, "nb = ", nb)

        accuracy = accuracy + int(float(na)) * p_a + int(float(nb)) * p_b
        na_total = na_total + int(float(na))
        nb_total = nb_total + int(float(nb))

        

    acc1 = accuracy/(na_total + nb_total)

    print("num_pairs = ", num_pairs)
    
    print("total plays = ", (na_total + nb_total))

    rmse = math.sqrt(rmse/(na_total + nb_total))

    upsets = upsets/(na_total + nb_total)
        
    print("accuracy = ", acc1)

    print("rmse = ", rmse)

    print("upsets = ", upsets)

    pred_accuracy = pred_accuracy/(na_total + nb_total)

    print("pred_accuracy = ", pred_accuracy)

    np.savetxt("prob_GNNRank.txt", prob_GNN)
    np.savetxt("prob_test_GNNRank.txt", prob_test)
    
    return acc1























