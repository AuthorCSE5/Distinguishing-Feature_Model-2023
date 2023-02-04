import numpy as np
import math
from scipy.special import comb
from matplotlib import pyplot as plt
#from statsmodels import api
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#import SF_model_weights
import gen_embedding
#import SFmodel_Data
from sklearn.model_selection import GridSearchCV


def SF_MLE_SFdata(embedding_obj, num_items, l, m, dim, features):

    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)
    U = np.copy(features)

    #prob_data_orig = np.copy(prob_data)

    #score_orig = prob_data_orig[:,num_items]

    score_orig = np.copy(embedding_obj.score)
    
    prob_test = np.copy(embedding_obj.prob_test)

    items_ranked_original = np.copy(embedding_obj.items_ranked_original)

    sign_data = train_set1[:,-1]
    y_train = np.copy(train_set1[:,-1])
    y_train[y_train == -1] = 0
    y_test = np.copy(test_set1[:,-1])
    test_sign = test_set1[:,-1]
    y_test[y_test == -1] = 0

    total_pairs = int(comb(num_items,2))

    print("m = ", m)
    s = train_set1.shape[0]
    print("s = ", s)
    
    X = np.zeros(((int(m * l)), int(dim)))
    y = np.zeros(int(m * l))
    

    #LR_l2 = {} #dictionary that conatins the L2 regularization parameters as they keys and the values are the coefficients/weights, scores and rankings. 

    for t in range(int(l * m)):
        
        for i in range(num_items-1):
            for j in range(i+1, num_items):
                if (train_set1[t][i] == 1 and train_set2[t][j] == 1) or (train_set1[t][j] == 1 and train_set2[t][i] == 1):
                    diff = np.abs(U[:,i] - U[:,j])
                    diff2 = np.amax(diff)
                    ind = np.argmax(diff)
                    if train_set1[t][num_items] == 1:
                        X[t,ind] = U[ind,i] - U[ind,j]
                        y[t] = 1
                    elif train_set1[t][num_items] == -1:
                        X[t,ind] = U[ind,i] - U[ind,j]
                        y[t] = 0
                
        
    
    
    lr = LogisticRegression(penalty = 'none', tol = 1e-10, C = 1, dual=False, fit_intercept = False, max_iter = 1000, verbose = 2, solver = 'sag').fit(X, y)
    w_est = lr.coef_ #weights
    #print(w_est)
    w_est = np.array(w_est)
    print(w_est)
    w_est = w_est/np.linalg.norm(w_est,1)
    print(w_est)

    

    #prediction accuracy on test data

    pred_accuracy = 0
    error = 0
    
    prob_SF_estimated = np.zeros((num_items, (num_items+1)))

    for i in range(num_items-1):
        for j in range(i+1, num_items):
    
            diff = np.abs(U[:,i] - U[:,j])
            diff2 = np.amax(diff)
            ind = np.argmax(diff)
            diff3 = (U[ind,i] - U[ind,j])
            
            #Copeland Scores
            prob_SF_estimated[i][j] = 1/(1+np.exp(-diff3 * w_est[0,ind]))
            prob_SF_estimated[j][i] = 1 - prob_SF_estimated[i][j]

            if prob_SF_estimated[i][j] > 0.5:
               prob_SF_estimated[i][num_items] = prob_SF_estimated[i][num_items] + 1

            elif prob_SF_estimated[i][j] < 0.5:
                prob_SF_estimated[j][num_items] = prob_SF_estimated[j][num_items] + 1
            

    
    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    s = test_data1.shape[0]

    rmse = 0

    for z in range(s):
        ind1 = np.argmax(test_data1[z,:])
        ind2 = np.argmax(test_data2[z,:])

        #print("size of prob_test = ", np.shape(prob_test))
        
        if (prob_SF_estimated[ind1][ind2] - 0.5) * (prob_test[z] - 0.5) > 0:
            pred_accuracy = pred_accuracy + 1
        else:
            error = error + 1

        rmse = rmse + (prob_SF_estimated[ind1][ind2] - prob_test[z])**2

    rmse = math.sqrt(rmse/s)

    print("rmse = ",rmse)

    pred_accuracy = pred_accuracy/s

    score = np.zeros(num_items)
    keys = np.zeros(num_items)

    for i in range(num_items):
        score[i] = prob_SF_estimated[i][num_items] #np.dot(U[:,i], np.transpose(w_est))
        keys[i] = i


    d = dict(zip(keys, score))

   

    items_ranked_sf = []
    score_sorted_sf = []

    for k in sorted(d, key=d.get, reverse=True):
        items_ranked_sf.append(k)
        score_sorted_sf.append(d[k])

    #print("items_ranked_DFLearn = ", items_ranked_dflearn)

    kt = stats.kendalltau(items_ranked_sf,items_ranked_original) #kendall tau correlation of 2 ranks

    A = 0
    D = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if ((items_ranked_original.tolist()).index(i) - (items_ranked_original.tolist()).index(j)) * (items_ranked_sf.index(i) - items_ranked_sf.index(j)) > 0 :
                A = A + 1
            else:
                D = D + 1
                
    ktc = (A - D)/total_pairs

    ktc2, p_val = stats.kendalltau(score,score_orig)
    print("ktc2 = ",ktc2)


    print("SF_ktc = ", ktc)
    print("SF_accuracy = ", pred_accuracy)

    return pred_accuracy, ktc2, rmse


def SF_MLE_DFdata(embedding_obj, num_items, l, m, dim):

    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)

    #prob_data_orig = np.copy(prob_data)

    #score_orig = prob_data_orig[:,num_items]
    
    score_orig = np.copy(embedding_obj.score)

    prob_test = np.copy(embedding_obj.prob_test)
    rank_data12 = np.copy(embedding_obj.rank_data12)
    rank_data22 = np.copy(embedding_obj.rank_data22)

    items_ranked_original = np.copy(embedding_obj.items_ranked_original)

    total_pairs = int(comb(num_items,2))

    print("m = ", m)
    s = train_set1.shape[0]
    print("s = ", s)
    
    X = np.zeros(((int(m * l)), int(num_items)))
    y = np.zeros(int(m * l))
    #X = np.zeros(((int(lm)), int(dim)))
    #y = np.zeros(int(lm))

    diff_total_train = np.abs(train_set1[:,:-1] - train_set2[:,:])
    diff_total_test = np.abs(test_set1[:,:-1] - test_set2[:,:])

    diff_rank = np.abs(rank_data12[:,:] - rank_data22[:,:])
    
    #LR_l2 = {} #dictionary that conatins the L2 regularization parameters as they keys and the values are the coefficients/weights, scores and rankings. 

    for t in range(int(l * m)):
        
        for i in range(num_items-1):
            for j in range(i+1, num_items):
                if (train_set1[t][i] == 1 and train_set2[t][j] == 1) or (train_set1[t][j] == 1 and train_set2[t][i] == 1):
                    
                    diff = diff_total_train[t,:]
                    
                    ind = np.argmax(diff)
                    
                    if train_set1[t][num_items] == 1:
                        X[t,ind] = train_set1[t][ind] - train_set2[t][ind]
                        y[t] = 1
                    elif train_set1[t][num_items] == -1:
                        X[t,ind] = train_set1[t][ind] - train_set2[t][ind]
                        y[t] = 0

                #print(X)   
        #y[t] = train_set1[t][num_items]
                
    
    
    lr = LogisticRegression(penalty = 'l2', tol = 1e-10, C = 1, dual=False, fit_intercept = False, max_iter = 10000, verbose = 2, solver = 'sag').fit(X, y)
    w_est = lr.coef_ #weights
    #print(w_est)
    w_est = np.array(w_est)
    #print(w_est)

    print(w_est)
    w_est = w_est/np.linalg.norm(w_est,1)
    print(w_est)

    '''score = np.zeros(num_items)
    keys = np.zeros(num_items)

    for i in range(num_items):
        score[i] = w_est[0,i]
        keys[i] = i

    d = dict(zip(keys, score))

    

    items_ranked_sf = []
    score_sorted_sf = []

    for k in sorted(d, key=d.get, reverse=True):
        items_ranked_sf.append(k)
        score_sorted_sf.append(d[k])

    

    kt = stats.kendalltau(items_ranked_sf,items_ranked_original) #kendall tau correlation of 2 ranks

    A = 0
    D = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if ((items_ranked_original.tolist()).index(i) - (items_ranked_original.tolist()).index(j)) * (items_ranked_sf.index(i) - items_ranked_sf.index(j)) > 0 :
                A = A + 1
            else:
                D = D + 1
                
    ktc = (A - D)/total_pairs'''
    #prediction accuracy on test data

    pred_accuracy = 0
    error = 0
    #prob_SF_original = np.zeros((num_items, num_items))
    prob_SF_estimated = np.zeros((num_items, (num_items+1)))

    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    s = test_data1.shape[0]

    '''for t in range(s):
        
        ind1 = np.argmax(test_data1[t,:])
        ind2 = np.argmax(test_data2[t,:])
        
        
        diff = diff_total_test[t,:]
                
        ind = np.argmax(diff)

        diff3 = test_data1[t][ind] - test_data2[t][ind]

                  
        
        prob_SF_estimated[ind1][ind2] = 1/(1+np.exp(-diff3 * w_est[0,ind]))
        prob_SF_estimated[ind2][ind1] = 1 - prob_SF_estimated[ind1][ind2]'''

    #diff_train = np.unique(diff_total_train)
    #diff_total = np.append(diff_train, diff_total_test, axis = 0)

    s_rank = rank_data12.shape[0]
    
    ct = 0

    #for ct in range(s_rank):
    for i in range(num_items-1):
        for j in range(i+1, num_items):
    
            #diff3 = (U[i] - U[j])
    
            #diff = diff_total[ct,:]        
                              
            #ind = np.argmax(diff)

            #diff3 = test_data1[t][ind] - test_data2[t][ind]

            diff = diff_rank[ct,:]
            ind = np.argmax(diff)
            diff3 = rank_data12[ct][ind] - rank_data22[ct][ind]
            
            
            prob_SF_estimated[i][j] = 1/(1+np.exp(-diff3 * w_est[0,ind]))
            prob_SF_estimated[j][i] = 1 - prob_SF_estimated[i][j]

            if prob_SF_estimated[i][j] > 0.5:
               prob_SF_estimated[i][num_items] = prob_SF_estimated[i][num_items] + 1

            elif prob_SF_estimated[i][j] < 0.5:
                prob_SF_estimated[j][num_items] = prob_SF_estimated[j][num_items] + 1

            ct += 1

    
            
            
    rmse = 0

    for w in range(s):
        ind1 = np.argmax(test_data1[w,:])
        ind2 = np.argmax(test_data2[w,:])

        
        if (prob_SF_estimated[ind1][ind2] - 0.5) * (prob_test[w][0] - 0.5) > 0:
            pred_accuracy = pred_accuracy + 1
        else:
            error = error + 1

        rmse = rmse + (prob_SF_estimated[ind1][ind2] - prob_test[w])**2

    rmse = math.sqrt(rmse/s)

    print("rmse = ",rmse)

    pred_accuracy = pred_accuracy/s

    score = np.zeros(num_items)
    keys = np.zeros(num_items)

    for i in range(num_items):
        score[i] = w_est[0,i]
        keys[i] = i

    d = dict(zip(keys, score))

    

    items_ranked_sf = []
    score_sorted_sf = []

    for k in sorted(d, key=d.get, reverse=True):
        items_ranked_sf.append(k)
        score_sorted_sf.append(d[k])

    

    kt = stats.kendalltau(items_ranked_sf,items_ranked_original) #kendall tau correlation of 2 ranks

    A = 0
    D = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if ((items_ranked_original.tolist()).index(i) - (items_ranked_original.tolist()).index(j)) * (items_ranked_sf.index(i) - items_ranked_sf.index(j)) > 0 :
                A = A + 1
            else:
                D = D + 1
                
    ktc = (A - D)/total_pairs

    ktc2, p_val = stats.kendalltau(score,score_orig)
    print("ktc2 = ",ktc2)

    
    print("SF_ktc = ", ktc)
    print("SF_accuracy = ", pred_accuracy)

    return pred_accuracy, ktc2, rmse

def SF_Copeland_DFdata(embedding_obj, num_items, l, m, dim):

    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)

    score_orig = np.copy(embedding_obj.score)


    prob_test = np.copy(embedding_obj.prob_test)
    rank_data12 = np.copy(embedding_obj.rank_data12)
    rank_data22 = np.copy(embedding_obj.rank_data22)

    items_ranked_original = np.copy(embedding_obj.items_ranked_original)

    total_pairs = int(comb(num_items,2))

    print("m = ", m)
    s = train_set1.shape[0]
    print("s = ", s)
    
    X = np.zeros(((int(m * l)), int(num_items)))
    y = np.zeros(int(m * l))
    #X = np.zeros(((int(lm)), int(dim)))
    #y = np.zeros(int(lm))

    diff_total_train = np.abs(train_set1[:,:-1] - train_set2[:,:])
    diff_total_test = np.abs(test_set1[:,:-1] - test_set2[:,:])

    diff_rank = np.abs(rank_data12[:,:] - rank_data22[:,:])
    
    #LR_l2 = {} #dictionary that conatins the L2 regularization parameters as they keys and the values are the coefficients/weights, scores and rankings. 

    for t in range(int(l * m)):
        
        for i in range(num_items-1):
            for j in range(i+1, num_items):
                if (train_set1[t][i] == 1 and train_set2[t][j] == 1) or (train_set1[t][j] == 1 and train_set2[t][i] == 1):
                    
                    diff = diff_total_train[t,:]
                    
                    ind = np.argmax(diff)
                    
                    if train_set1[t][num_items] == 1:
                        X[t,ind] = train_set1[t][ind] - train_set2[t][ind]
                        y[t] = 1
                    elif train_set1[t][num_items] == -1:
                        X[t,ind] = train_set1[t][ind] - train_set2[t][ind]
                        y[t] = 0

                #print(X)   
        #y[t] = train_set1[t][num_items]
                
    
    
    lr = LogisticRegression(penalty = 'l2', tol = 1e-10, C = 1, dual=False, fit_intercept = False, max_iter = 10000, verbose = 2, solver = 'sag').fit(X, y)
    w_est = lr.coef_ #weights
    #print(w_est)
    w_est = np.array(w_est)
    #print(w_est)

    print(w_est)
    w_est = w_est/np.linalg.norm(w_est,1)
    #print(w_est)

    '''score = np.zeros(num_items)
    keys = np.zeros(num_items)

    for i in range(num_items):
        score[i] = w_est[0,i]
        keys[i] = i

    d = dict(zip(keys, score))

    

    items_ranked_sf = []
    score_sorted_sf = []

    for k in sorted(d, key=d.get, reverse=True):
        items_ranked_sf.append(k)
        score_sorted_sf.append(d[k])

    

    kt = stats.kendalltau(items_ranked_sf,items_ranked_original) #kendall tau correlation of 2 ranks

    A = 0
    D = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if ((items_ranked_original.tolist()).index(i) - (items_ranked_original.tolist()).index(j)) * (items_ranked_sf.index(i) - items_ranked_sf.index(j)) > 0 :
                A = A + 1
            else:
                D = D + 1
                
    ktc = (A - D)/total_pairs'''
    #prediction accuracy on test data

    pred_accuracy = 0
    error = 0
    #prob_SF_original = np.zeros((num_items, num_items))
    prob_SF_estimated = np.zeros((num_items, (num_items+1)))

    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    s = test_data1.shape[0]

    '''for t in range(s):
        
        ind1 = np.argmax(test_data1[t,:])
        ind2 = np.argmax(test_data2[t,:])
        
        
        diff = diff_total_test[t,:]
                
        ind = np.argmax(diff)

        diff3 = test_data1[t][ind] - test_data2[t][ind]

                  
        
        prob_SF_estimated[ind1][ind2] = 1/(1+np.exp(-diff3 * w_est[0,ind]))
        prob_SF_estimated[ind2][ind1] = 1 - prob_SF_estimated[ind1][ind2]'''

    #diff_train = np.unique(diff_total_train)
    #diff_total = np.append(diff_train, diff_total_test, axis = 0)

    s_rank = rank_data12.shape[0]
    
    ct = 0

    #for ct in range(s_rank):
    for i in range(num_items-1):
        for j in range(i+1, num_items):
    
            #diff3 = (U[i] - U[j])
    
            #diff = diff_total(ct,:)        
                              
            #ind = np.argmax(diff)

            #diff3 = test_data1[t][ind] - test_data2[t][ind]

            diff = diff_rank[ct,:]
            ind = np.argmax(diff)
            diff3 = rank_data12[ct][ind] - rank_data22[ct][ind]
            
            
            prob_SF_estimated[i][j] = 1/(1+np.exp(-diff3 * w_est[0,ind]))
            prob_SF_estimated[j][i] = 1 - prob_SF_estimated[i][j]

            if prob_SF_estimated[i][j] > 0.5:
               prob_SF_estimated[i][num_items] = prob_SF_estimated[i][num_items] + 1

            elif prob_SF_estimated[i][j] < 0.5:
                prob_SF_estimated[j][num_items] = prob_SF_estimated[j][num_items] + 1

            ct += 1
                
   

    print("prob_estimated = ", prob_SF_estimated)

    score = np.zeros(num_items)
    keys = np.zeros(num_items)

    for i in range(num_items):
        score[i] = prob_SF_estimated[i][num_items] #w_est[0,i]
        keys[i] = i

    d = dict(zip(keys, score))

    

    items_ranked_sf = []
    score_sorted_sf = []

    for k in sorted(d, key=d.get, reverse=True):
        items_ranked_sf.append(k)
        score_sorted_sf.append(d[k])

    

    kt = stats.kendalltau(items_ranked_sf,items_ranked_original) #kendall tau correlation of 2 ranks

    A = 0
    D = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if ((items_ranked_original.tolist()).index(i) - (items_ranked_original.tolist()).index(j)) * (items_ranked_sf.index(i) - items_ranked_sf.index(j)) > 0 :
                A = A + 1
            else:
                D = D + 1
                
    ktc = (A - D)/total_pairs

    ktc2, p_val = stats.kendalltau(score,score_orig)
    print("ktc2 = ",ktc2)

    upsets = 0
    rmse = 0

    for w in range(s):
        ind1 = np.argmax(test_data1[w,:])
        ind2 = np.argmax(test_data2[w,:])

        
        if (prob_SF_estimated[ind1][ind2] - 0.5) * (prob_test[w][0] - 0.5) > 0:
            pred_accuracy = pred_accuracy + 1
        else:
            error = error + 1

        rmse = rmse + (prob_SF_estimated[ind1][ind2] - prob_test[w])**2

    

        #upset calculation
        if (score[ind1] - score[ind2]) * (prob_test[w] - 0.5) < 0:
            upsets += 1

    pred_accuracy = pred_accuracy/s

    rmse = math.sqrt(rmse/s)
    print("rmse = ",rmse)

    upsets = upsets/s
    
    print("SF_ktc = ", ktc)
    print("SF_accuracy = ", pred_accuracy)
    print("SF_upsets = ", upsets)

    return pred_accuracy, ktc2, rmse


