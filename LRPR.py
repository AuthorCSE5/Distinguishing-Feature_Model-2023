import numpy as np
import math
from scipy.special import comb
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
#import matlab.engine
import OptSpace

def LRPR(embedding_obj, num_items, l , m):

    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)

    score_orig = np.copy(embedding_obj.score)


    prob_test = np.copy(embedding_obj.prob_test)

    #items_ranked_original = np.copy(embedding_obj.items_ranked_original)
    items_ranked_original = np.copy(embedding_obj.items_ranked_original)

    total_pairs = int(comb(num_items,2))


    A = np.zeros((num_items,num_items))
    P = np.zeros((num_items,num_items))


    t = np.zeros((num_items,num_items))
    
    train_size = train_set1.shape[0]
    

    for w in range(train_size):
        i = np.argmax(train_set1[w,:-1])
        j = np.argmax(train_set2[w,:])
        
        
        if train_set1[w][i] == 1 and train_set2[w][j] == 1:
            
            
            if train_set1[w][num_items] == 1: #i beats j
                t[i][j] = t[i][j] + 1
                
        
            elif train_set1[w][num_items] == -1: #j beats i
                t[j][i] = t[j][i] + 1
                
    t = (1/l) * t

    for i in range(num_items):

        P[i][i] = 0.5

        for j in range(i+1, num_items):
            
            #if t[i][j] > 0 or t[j][i] > 0:
            P[i][j] = t[i][j] #i beats j
            P[j][i] = 1 - t[i][j]


    #Apply Link Function
    #Logit Link

    LP = np.zeros((num_items, num_items))
    

    for i in range(num_items):
        for j in range(i+1, num_items):
            if P[i][j] > 0 and (1-P[i][j]) > 0:
               LP[i][j] = math.log2(P[i][j]) - math.log2(1-(P[i][j]))
               LP[j][i] = math.log2(P[j][i]) - math.log2(1-(P[j][i]))

    #Apply Matrix Completion routine OPTSPACE

    #eng = matlab.engine.start_matlab()
    #eng.edit('OptSpace',nargout=0)

    tol = 1e-4
    niter = 100

    #LP2 = LP.astype(float)
    
    #X, S, Y, dist = eng.OptSpace(LP2,2.0,[],tol)

    X, S, Y, dist = OptSpace.OptSpace(LP, 2, niter, tol)
    
    XSYprime = np.dot(np.dot(X,S),np.transpose(Y))
    
    LP_MC = XSYprime

    print("LP_MC = ", LP_MC)
    
    #Apply inverse link transform

    P2 = np.zeros((num_items,num_items))

    eps = 0

    for i in range(num_items):
        for j in range(num_items):
            if (i != j and LP_MC[i][j] > LP_MC[j][i]):
                P2[i][j] = 0.5 + min(abs(np.exp(LP_MC[i][j] + eps)/(1+np.exp(LP_MC[i][j] + eps)) - 0.5), abs(np.exp(LP_MC[j][i] + eps)/(1 + np.exp(LP_MC[j][i] + eps)) - 0.5))
            elif (i != j and LP_MC[i][j] < LP_MC[j][i]):
                P2[i][j] = 0.5 - min(abs(np.exp(LP_MC[i][j] + eps)/(1+np.exp(LP_MC[i][j] + eps)) - 0.5), abs(np.exp(LP_MC[j][i] + eps)/(1 + np.exp(LP_MC[j][i] + eps)) - 0.5))
            elif i == j:
                P2[i][j] = 0.5
    #Copeland Ranking

    score = np.zeros(num_items)
    
    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if P2[i][j] > 0.5:
                score[i] += 1
            elif P2[j][i] > 0.5:
                score[j] += 1

    print("score = ", score)

    print("P2 = ", P2)

    items = np.zeros(num_items)

    for i in range(num_items):
        items[i] = i


    
    dictionary = dict(zip(items, score))


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

    ktc2, p_val = stats.kendalltau(score,score_orig)
    print("ktc2 = ",ktc2)

    #print("items_ranked = ", items_ranked)

    #ktc2, p_val = stats.kendalltau(items_ranked,items_ranked_original)

    #prediction accuracy on test data

    pred_accuracy = 0
    error = 0
    upsets = 0
    rmse = 0
    
    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    s = test_data1.shape[0]
    #print("s = ", s)

    for w in range(s):
        #print("w = ", w)
        ind1 = np.argmax(test_data1[w,:])
        ind2 = np.argmax(test_data2[w,:])
        
        if (P2[ind1][ind2] - 0.5) * (prob_test[w] - 0.5) > 0:
            pred_accuracy = pred_accuracy + 1
        else:
            error = error + 1

        if (score[ind1] - score[ind2]) * (prob_test[w] - 0.5) < 0:
            upsets += 1

        rmse = rmse + (P2[ind1][ind2] - prob_test[w])**2

    pred_accuracy = pred_accuracy/s

    upsets = upsets/s

    rmse = math.sqrt(rmse/s)
    print("rmse = ", rmse)

    print("prediction accuracy = ", pred_accuracy)
    print("upsets = ", upsets)
       
    return pred_accuracy, ktc2, rmse


def LRPR_RealData(train_data1, train_data2, test_data1, test_data2, num_items, prob_data):

    train_set1 = np.copy(train_data1)
    train_set2 = np.copy(train_data2)
    test_set1 = np.copy(test_data1)
    test_set2 = np.copy(test_data2)

    
    prob_data = np.copy(prob_data)

    #items_ranked_original = np.copy(embedding_obj.items_ranked_original)
    
    total_pairs = int(comb(num_items,2))

    P = np.zeros((num_items,num_items))


    t = np.zeros((num_items,num_items))
    
    train_size = train_set1.shape[0]
        

    for w in range(train_size):
        i = np.argmax(train_set1[w,:-1])
        j = np.argmax(train_set2[w,:])
        
        
        if train_set1[w][i] == 1 and train_set2[w][j] == 1:
            
            
            if train_set1[w][num_items] == 1: #i beats j
                t[i][j] = t[i][j] + 1
                                
        
            elif train_set1[w][num_items] == -1: #j beats i
                t[j][i] = t[j][i] + 1
                
        
    for i in range(num_items):
        P[i][i] = 0.5
        for j in range(i+1, num_items):
            if t[i][j] + t[j][i] > 0:
                P[i][j] = (t[i][j] + 1)/(t[i][j] + t[j][i] + 2) #i beats j
                P[j][i] = 1 - P[i][j]
        
                

    #Apply Link Function
    #Logit Link

    LP = np.zeros((num_items, num_items))
    eps = 0.001
    print("eps = ", eps)

    for i in range(num_items):
        for j in range(i+1, num_items):
            if P[i][j] == 1:
               P[i][j] = 1 - eps
               P[j][i] = eps
            if P[i][j] > 0 and (1-P[i][j]) > 0:
               LP[i][j] = math.log2(P[i][j]) - math.log2(1-(P[i][j]))
               LP[j][i] = math.log2(P[j][i]) - math.log2(1-(P[j][i]))

    #Apply Matrix Completion routine OPTSPACE

    #eng = matlab.engine.start_matlab()
    #eng.edit('OptSpace',nargout=0)

    tol = 1e-4
    niter = 200

    #LP2 = LP.astype(float)
    
    #X, S, Y, dist = eng.OptSpace(LP2,2.0,[],tol)

    X, S, Y, dist = OptSpace.OptSpace(LP, 2, niter, tol)
    
    XSYprime = np.dot(np.dot(X,S),np.transpose(Y))
    
    LP_MC = XSYprime

    print("LP_MC = ", LP_MC)
    
    #Apply inverse link transform

    P2 = np.zeros((num_items,num_items))

    #eps = 0.01

    for i in range(num_items):
        for j in range(num_items):
            if (i != j and LP_MC[i][j] > LP_MC[j][i]):
                P2[i][j] = 0.5 + min(abs(np.exp(LP_MC[i][j] + eps)/(1+np.exp(LP_MC[i][j] + eps)) - 0.5), abs(np.exp(LP_MC[j][i] + eps)/(1 + np.exp(LP_MC[j][i] + eps)) - 0.5))
            elif (i != j and LP_MC[i][j] < LP_MC[j][i]):
                P2[i][j] = 0.5 - min(abs(np.exp(LP_MC[i][j] + eps)/(1+np.exp(LP_MC[i][j] + eps)) - 0.5), abs(np.exp(LP_MC[j][i] + eps)/(1 + np.exp(LP_MC[j][i] + eps)) - 0.5))
            elif i == j:
                P2[i][j] = 0.5

    '''#Copeland Ranking

    score = np.zeros(num_items)
    
    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if P2[i][j] > 0.5:
                score[i] += 1
            else:
                score[j] += 1

    print("score = ", score)

    items = np.zeros(num_items)

    for i in range(num_items):
        items[i] = i


    
    dictionary = dict(zip(items, score))


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

    #ktc2, p_val = stats.kendalltau(items_ranked,items_ranked_original)'''

    #prediction accuracy on test data

    pred_accuracy = 0
    error = 0
    rmse = 0
    
    
    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    s = test_data1.shape[0]
    #print("s = ", s)

    print("prob_data", prob_data)
    print("P2", P2)

    pred = np.zeros(s)

    for w in range(s):
        #print("w = ", w)
        ind1 = np.argmax(test_data1[w,:])
        ind2 = np.argmax(test_data2[w,:])

        pred[w] = P2[ind1][ind2]
        
        if (P2[ind1][ind2] - 0.5) * (prob_data[ind1][ind2] - 0.5) > 0:
            pred_accuracy = pred_accuracy + 1
        else:
            error = error + 1

        rmse = rmse + (P2[ind1][ind2] - prob_data[ind1][ind2])**2

    pred_accuracy = pred_accuracy/s
    rmse = math.sqrt(rmse/s)

    print("prediction accuracy = ", pred_accuracy)

    print("rmse = ", rmse)

    #np.savetxt("LRPR2_prediction_dota.txt", P2)

       
    return pred_accuracy, rmse
                
                

    
        

                


    

