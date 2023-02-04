import numpy as np
#np.random.seed(0)
import math
from scipy.special import comb
from scipy.linalg import eig
#from sympy import Matrix
import matplotlib.pyplot as plt
from scipy import stats
from itertools import chain


def RankCentrality(train_data1, train_data2, test_data1, test_data2, num_items, prob_data):

    train_set1 = np.copy(train_data1)
    train_set2 = np.copy(train_data2)
    test_set1 = np.copy(test_data1)
    test_set2 = np.copy(test_data2)

    
    prob_data = np.copy(prob_data)

    #items_ranked_original = np.copy(embedding_obj.items_ranked_original)

    total_pairs = int(comb(num_items,2))

    A = np.zeros((num_items,num_items))
    P = np.zeros((num_items,num_items))


    t = np.zeros((num_items,num_items))
    d = np.zeros(num_items) #degree of each item

    train_size = train_set1.shape[0]
    count_data = np.zeros((num_items, num_items))

    for w in range(train_size):
        i = np.argmax(train_set1[w,:-1])
        j = np.argmax(train_set2[w,:])
        
                
        #print("w = ", w)
        if train_set1[w][i] == 1 and train_set2[w][j] == 1:
            
            
            if train_set1[w][num_items] == 1: #j beats i
                t[j][i] = t[j][i] + 1
                d[j] = d[j] + 1
                d[i] = d[i] + 1
                #count_data[i][j] += 1
        
            elif train_set1[w][num_items] == -1: #i beats j
                t[i][j] = t[i][j] + 1
                d[i] = d[i] + 1
                d[j] = d[j] + 1
                #count_data[j][i] += 1

        #if count_data[i][j] + count_data[j][i] > 0:
         #   t[j][i] = t[j][i]/(count_data[i][j] + count_data[j][i])
          #  t[i][j] = t[i][j]/(count_data[i][j] + count_data[j][i])
           # d[j] = d[j]/(count_data[i][j] + count_data[j][i])
            #d[i] = d[i]/(count_data[i][j] + count_data[j][i])
                                                


    #t = (1/l) * t
    #d = (1/l) * d

    #print(t)

    d_max = np.amax(d)
    #print("d = ", d, "d_max = ", d_max)

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            
            #if t[i][j] > 0 or t[j][i] > 0:
            A[i][j] = (t[i][j]+1)/(t[i][j] + t[j][i]+2) #j beats i
            A[j][i] = 1 - A[i][j]



    for i in range(num_items):
        for j in range(i+1, num_items):
            #if i != j:
            P[i][j] = (1/d_max)*A[i][j] #j beats i
            P[j][i] = (1/d_max)*A[j][i]
        #if i == j:
        P[i][i] = 1 - sum(P[i][:])

    #print(P)

    max_iter = 30
    epsilon = 0.001

    estimates = (1/num_items) * np.ones(num_items);
    estimates_old = estimates;

    convergence_iter = num_items;
    #print("estimates = ", estimates)



    P2 = P.transpose()
    #print(P2)
    iteration = []
    error_hr = []


    eigval, eigvec = eig(P2)
    
    eigval = np.real(eigval)
        
    max_eigval = np.max(eigval)

    eig_vec_index = np.argmax(eigval)
    
    ev = eigvec[:,eig_vec_index]

    estimates = ev/np.sum(ev)
    
    ranking = np.argsort(-estimates)

        
    '''for i in range(max_iter):

        iteration.append(i)
        #print("estimates_old = ", estimates)
        estimates = P2.dot(estimates)
        #print("estimates_new = ", estimates)
        estimates = estimates/np.linalg.norm(estimates,1)
        
         
        #error_hr.append(np.linalg.norm(score_original-estimates,2))
        
        #if(np.linalg.norm(estimates, 1))< epsilon * np.linalg.norm(estimates_old, 1):
        if(np.linalg.norm(estimates-estimates_old, 2))< epsilon * np.linalg.norm(estimates_old, 2):
            convergence_iter = i
            print(convergence_iter)
            break
        
        if(i == max_iter):
            print("Heuristic failed to converge")
        
        estimates_old = estimates'''

    

    items = np.zeros(num_items)

    for i in range(num_items):
        items[i] = i


    #print("estimates = ", estimates)
    
    '''dictionary = dict(zip(items, estimates))


    items_ranked = []
    score_sorted = []

    for k in sorted(dictionary, key=dictionary.get, reverse=True):
        items_ranked.append(k)
        score_sorted.append(dictionary[k])

    #ktc, p_val = stats.kendalltau(items_ranked,items_ranked_original)

    A = 0
    D = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if ((items_ranked_original.tolist()).index(i) - (items_ranked_original.tolist()).index(j)) * (items_ranked.index(i) - items_ranked.index(j)) > 0 :
                A = A + 1
            else:
                D = D + 1
                
    ktc = (A - D)/total_pairs
    print("ktc = ",ktc)

    #print("items_ranked = ", items_ranked)

    ktc2, p_val = stats.kendalltau(items_ranked,items_ranked_original)'''

    #prediction accuracy on test data

    pred_accuracy = 0
    error = 0
    rmse = 0
    prob_RC = np.zeros((num_items, num_items))

    for i in range(0, num_items):
        for j in range(i+1, num_items):
            prob_RC[i][j] = estimates[i]/(estimates[i] + estimates[j]) #1/(1+np.exp(-(estimates[i]-estimates[j])))
            prob_RC[j][i] = 1 - prob_RC[i][j]

    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    winners = np.copy(test_set1[:,-1])

    print("winners = ", winners)
    
    s = test_data1.shape[0]
    print("s = ", s)

    ind_test = np.zeros((s,2))
    

    for t in range(s):

        
        ind1 = np.argmax(test_data1[t,:])
        ind2 = np.argmax(test_data2[t,:])
        ind_test[t,0] = ind1
        ind_test[t,1] = ind2
        

    test_index2 = np.unique(ind_test, axis = 0)
    
    print("ind_test = ", ind_test.shape[0])
    print("text_index2 = ", test_index2.shape[0])
    
    s2 = test_index2.shape[0]
    print("s2 = ",s2)

    p_a = 0
    p_b = 0
    accuracy = 0

    for t in range(s):

        
        #ind1 = int(test_index2[t,0])
        #ind2 = int(test_index2[t,1])
        ind1 = np.argmax(test_data1[t,:])
        ind2 = np.argmax(test_data2[t,:])

        #print("test_index2[t,0] = ", test_index2[t,0])
        #print("test_index2[t,1] = ", test_index2[t,1])
    
        if ((prob_data[ind1][ind2] - 0.5)*(prob_RC[ind1][ind2] - 0.5)) > 0:
            pred_accuracy = pred_accuracy + 1
        else:
            error = error + 1

        rmse = rmse + (prob_RC[ind1][ind2] - prob_data[ind1][ind2])**2

        if (prob_RC[ind1][ind2] >= 0.5 and winners[t] == 1) or (prob_RC[ind1][ind2] < 0.5 and winners[t] == -1):
            accuracy += 1


    '''s = test_data1.shape[0]

    rmse = 0
    pred_accuracy = 0
    accuracy = 0

    for t in range(s):

        ind1 = np.argmax(test_data1[t,:])
        ind2 = np.argmax(test_data2[t,:])

        if ((prob_data[ind1][ind2] - 0.5)*(prob_RC[ind1][ind2] - 0.5)) > 0:
            pred_accuracy = pred_accuracy + 1
        
        rmse = rmse + (prob_RC[ind1][ind2] - prob_data[ind1][ind2])**2

        if (prob_RC[ind1][ind2] >= 0.5 and winners[t] == 1) or (prob_RC[ind1][ind2] < 0.5 and winners[t] == -1):
            accuracy += 1'''

        

    pred_accuracy = pred_accuracy/s

    print("pred_accuracy = ", (pred_accuracy))

    accuracy = accuracy/s

    print("accuracy = ", accuracy)

    rmse = math.sqrt(rmse/s)

    print("rmse = ", rmse)

    #pred_accuracy = pred_accuracy/s2

    #print("pred_accuracy = ", (pred_accuracy))
    
    return pred_accuracy, rmse


def RankCentrality_SFData(embedding_obj, num_items, l , m):

    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)


    prob_test = np.copy(embedding_obj.prob_test)

    items_ranked_original = np.copy(embedding_obj.items_ranked_original)

    total_pairs = int(comb(num_items,2))

    A = np.zeros((num_items,num_items))
    P = np.zeros((num_items,num_items))


    t = np.zeros((num_items,num_items))
    d = np.zeros(num_items) #degree of each item

    for w in range(m*l):
        for j in range(num_items-1):
            if train_set1[w][j] == 1:
                for k in range(j+1, num_items):
                
                    if train_set2[w][k] == 1:
                        if train_set1[w][num_items] == 1: #j beats k
                            t[k][j] = t[k][j] + 1
                            d[k] = d[k] + 1
                            d[j] = d[j] + 1
                    
                        elif train_set1[w][num_items] == -1: #k beats j
                            t[j][k] = t[j][k] + 1
                            d[k] = d[k] + 1
                            d[j] = d[j] + 1
                    


    t = (1/l) * t
    d = (1/l) * d

    #print(t)

    d_max = np.amax(d)
    #print("d = ", d, "d_max = ", d_max)

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            
            if t[i][j] > 0 or t[j][i] > 0:
               A[i][j] = t[i][j]/(t[i][j] + t[j][i]) #j beats i
               A[j][i] = 1 - A[i][j]



    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if i != j:
                P[i][j] = (1/d_max)*A[i][j] #j beats i
                P[j][i] = (1/d_max)*A[j][i]
        if i == j:
            P[i][j] = 1 - sum(P[i][:])

    #print(P)

    max_iter = 30
    epsilon = 0.001

    estimates = (1/num_items) * np.ones(num_items);
    estimates_old = estimates;

    convergence_iter = num_items;
    #print("estimates = ", estimates)



    P2 = P.transpose()
    #print(P2)
    iteration = []
    error_hr = []



    for i in range(max_iter):

        iteration.append(i)
        #print("estimates_old = ", estimates)
        estimates = P2.dot(estimates)
        #print("estimates_new = ", estimates)
        estimates = estimates/np.linalg.norm(estimates,1)
        
         
        #error_hr.append(np.linalg.norm(score_original-estimates,2))
        
        #if(np.linalg.norm(estimates, 1))< epsilon * np.linalg.norm(estimates_old, 1):
        if(np.linalg.norm(estimates-estimates_old, 2))< epsilon * np.linalg.norm(estimates_old, 2):
            convergence_iter = i
            print(convergence_iter)
            break
        
        if(i == max_iter):
            print("Heuristic failed to converge")
        
        estimates_old = estimates

    items = np.zeros(num_items)

    for i in range(num_items):
        items[i] = i


    print("estimates = ", estimates)
    
    dictionary = dict(zip(items, estimates))


    items_ranked = []
    score_sorted = []

    for k in sorted(dictionary, key=dictionary.get, reverse=True):
        items_ranked.append(k)
        score_sorted.append(dictionary[k])

    #ktc, p_val = stats.kendalltau(items_ranked,items_ranked_original)

    A = 0
    D = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if ((items_ranked_original.tolist()).index(i) - (items_ranked_original.tolist()).index(j)) * (items_ranked.index(i) - items_ranked.index(j)) > 0 :
                A = A + 1
            else:
                D = D + 1
                
    ktc = (A - D)/total_pairs
    print("ktc = ",ktc)

    #print("items_ranked = ", items_ranked)

    ktc2, p_val = stats.kendalltau(items_ranked,items_ranked_original)

    #prediction accuracy on test data

    pred_accuracy = 0
    error = 0
    prob_RC = np.zeros((num_items, num_items))

    for i in range(0, num_items):
        for j in range(i+1, num_items):
            prob_RC[i][j] = 1/(1+np.exp(-(estimates[i]-estimates[j])))
            prob_RC[j][i] = 1 - prob_RC[i][j]

    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    s = test_data1.shape[0]
    #print("s = ", s)

    rmse = 0
    
    for w in range(s):
        #print("w = ", w)
        ind1 = np.argmax(test_data1[w,:])
        ind2 = np.argmax(test_data2[w,:])
        
        if (prob_RC[ind1][ind2] - 0.5) * (prob_test[w] - 0.5) > 0:
            pred_accuracy = pred_accuracy + 1
        else:
            error = error + 1

        rmse = rmse + (prob_RC[ind1][ind2] - prob_test[w])**2

    pred_accuracy = pred_accuracy/s

    rmse = math.sqrt(rmse/s)

    print("prediction accuracy = ", pred_accuracy)

    print("rmse = ", rmse)
    
    return pred_accuracy, ktc























