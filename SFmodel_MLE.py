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


def SF_MLE_SFdata(embedding_obj, num_items, l, m, dim, embed):

    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)
    U = np.copy(embed)

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
                if train_set1[t][i] == 1 and train_set2[t][j] == 1:
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
    #print(w_est)

    score = np.zeros(num_items)
    keys = np.zeros(num_items)

    for i in range(num_items):
        score[i] = np.dot(U[:,i], np.transpose(w_est))
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

    #prediction accuracy on test data

    pred_accuracy = 0
    error = 0
    
    prob_SF_estimated = np.zeros((num_items, num_items))

    for i in range(num_items-1):
        for j in range(i+1, num_items):
    
            diff = np.abs(U[:,i] - U[:,j])
            diff2 = np.amax(diff)
            ind = np.argmax(diff)
            diff3 = (U[ind,i] - U[ind,j])
            
            
            prob_SF_estimated[i][j] = 1/(1+np.exp(-diff3 * w_est[0,ind]))
            prob_SF_estimated[j][i] = 1 - prob_SF_estimated[i][j]
            

    
    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    s = test_data1.shape[0]

   

    for z in range(s):
        ind1 = np.argmax(test_data1[z,:])
        ind2 = np.argmax(test_data2[z,:])

        #print("size of prob_test = ", np.shape(prob_test))
        
        if (prob_SF_estimated[ind1][ind2] - 0.5) * (prob_test[z] - 0.5) > 0:
            pred_accuracy = pred_accuracy + 1
        else:
            error = error + 1

    pred_accuracy = pred_accuracy/s

    return pred_accuracy, ktc


def SF_MLE_DFdata(embedding_obj, num_items, l, m, dim):

    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)


    prob_test = np.copy(embedding_obj.prob_test)

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
    
    #LR_l2 = {} #dictionary that conatins the L2 regularization parameters as they keys and the values are the coefficients/weights, scores and rankings. 

    for t in range(int(l * m)):
        
        for i in range(num_items-1):
            for j in range(i+1, num_items):
                if train_set1[t][i] == 1 and train_set2[t][j] == 1:
                    
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
    #prediction accuracy on test data

    pred_accuracy = 0
    error = 0
    #prob_SF_original = np.zeros((num_items, num_items))
    prob_SF_estimated = np.zeros((num_items, num_items))

    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    s = test_data1.shape[0]

    for t in range(s):
        for i in range(num_items-1):
            for j in range(i+1, num_items):
        
                #diff3 = (U[i] - U[j])
                diff = diff_total_test[t,:]
                        
                ind = np.argmax(diff)

                diff3 = test_data1[t][ind] - test_data2[t][ind]
                
                
                prob_SF_estimated[i][j] = 1/(1+np.exp(-diff3 * w_est[0,ind]))
                prob_SF_estimated[j][i] = 1 - prob_SF_estimated[i][j]
                
   

    for w in range(s):
        ind1 = np.argmax(test_data1[w,:])
        ind2 = np.argmax(test_data2[w,:])

        
        if (prob_SF_estimated[ind1][ind2] - 0.5) * (prob_test[w][0] - 0.5) > 0:
            pred_accuracy = pred_accuracy + 1
        else:
            error = error + 1

    pred_accuracy = pred_accuracy/s

     
    
    print("SF_ktc = ", ktc)
    print("SF_accuracy = ", pred_accuracy)

    return pred_accuracy, ktc




def SF_MLE_BTLdata(embedding_obj, num_items, l, m, dim):

    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)


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
    
    diff_total_train = np.abs(train_set1[:,:-1] - train_set2[:,:])
    diff_total_test = np.abs(test_set1[:,:-1] - test_set2[:,:])
    
    #LR_l2 = {} #dictionary that conatins the L2 regularization parameters as they keys and the values are the coefficients/weights, scores and rankings. 

    for t in range(int(l * m)):
        
        for i in range(num_items-1):
            for j in range(i+1, num_items):
                
                diff = diff_total_train[t,:]
                ind = np.argmax(diff)
                
                if train_set1[t][i] == 1 and train_set2[t][j] == 1:
                    
                    if train_set1[t][num_items] == 1:
                        #X[t,0] = U[i] - U[j]
                        X[t,ind] = train_set1[t][ind] - train_set2[t][ind]
                        y[t] = 1
                    elif train_set1[t][num_items] == -1:
                        #X[t,0] = U[i] - U[j]
                        X[t,ind] = train_set1[t][ind] - train_set2[t][ind]
                        y[t] = 0

                #print(X)   
        #y[t] = train_set1[t][num_items]
                
    
    
    lr = LogisticRegression(penalty = 'l2', tol = 1e-10, C = 1, dual=False, fit_intercept = False, max_iter = 10000, verbose = 2, solver = 'sag').fit(X, y)
    w_est = lr.coef_ #weights
    print("w_est = ", w_est)
    w_est = np.array(w_est)
    #print(w_est)

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
    #prediction accuracy on test data

    pred_accuracy = 0
    error = 0
    #prob_SF_original = np.zeros((num_items, num_items))
    prob_SF_estimated = np.zeros((num_items, num_items))

    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    s = test_data1.shape[0]

    for t in range(s):
        for i in range(num_items-1):
            for j in range(i+1, num_items):
        
                #diff3 = (U[i] - U[j])
                diff = diff_total_test[t,:]
                        
                ind = np.argmax(diff)

                diff3 = test_data1[t][ind] - test_data2[t][ind]
                
                
                prob_SF_estimated[i][j] = 1/(1+np.exp(-diff3 * w_est[0,ind]))
                prob_SF_estimated[j][i] = 1 - prob_SF_estimated[i][j]
                
   

    for w in range(s):
        ind1 = np.argmax(test_data1[w,:])
        ind2 = np.argmax(test_data2[w,:])

        
        if (prob_SF_estimated[ind1][ind2] - 0.5) * (prob_test[w][0] - 0.5) > 0:
            pred_accuracy = pred_accuracy + 1
        else:
            error = error + 1

    pred_accuracy = pred_accuracy/s

     
    
    print("SF_ktc = ", ktc)
    print("SF_accuracy = ", pred_accuracy)

    return pred_accuracy, ktc


