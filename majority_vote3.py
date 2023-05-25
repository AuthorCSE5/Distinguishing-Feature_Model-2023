import numpy as np
import pandas as pd
import math
from scipy.special import comb
from matplotlib import pyplot as plt
#from statsmodels import api
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#import SF_model_weights
import gen_embedding
import create_data
import create_data3
#import SFmodel_Data
from sklearn.model_selection import GridSearchCV


def majority_vote_prob(a,b,params,dim,num_items):  

    a = int(float(a))
    b = int(float(b))
    
    mu_a = params[0:dim,a]
    mu_b = params[0:dim,b]

    p_val = np.zeros(dim)

    for i in range(dim):
        
        p_val[i] = 1 - norm.cdf(0, mu_a[i] - mu_b[i], 1/math.sqrt(1))

    w_ab = p_val[0]*p_val[1]*p_val[2] + (1 - p_val[0])*p_val[1]*p_val[2] + p_val[0]*(1 - p_val[1])*p_val[2] + p_val[0]*p_val[1]*(1 - p_val[2])
    w_ba = 1 - w_ab
    #print("w_ab = ", w_ab )
    #print("math.log2(w_ab) = ", math.log2(w_ab))

    return w_ab

def grad_descent(params, data2, n_iter, learning_rate, dim, num_items, num_pairs):

    params2 = params.reshape(dim,num_items)  #generate an initial point, here the parameters generated are passed

    eps = learning_rate * 0.1

    for i in range(n_iter):

        #calculate gradient

        #f, gradf = majority_vote_ll(params, data2, dim, num_items, num_pairs, 'train')

        f, gradf = obj_fn(params, data2)

        print("train_ll = ", f, "i = ", i)#, "gradient = ", gradf)#, "gradf = ", gradf, "parameters = ", params)

        #update the parameters

        params = params + learning_rate * gradf

        #learning_rate = (1 - eps) * learning_rate

    return params        

    

def majority_vote_ll(params, data, dim, num_items, num_pairs, flag):

    data = np.reshape(data, (num_pairs,4))
    #params = params.reshape(dim,num_items)
    
    ll = 0
    fn_der = np.zeros((dim,num_items))

    prob_testdata = np.zeros((num_items, num_items))
    prob_estimated = np.zeros((num_items, num_items))

    rmse = 0
    pred_accuracy = 0

    na_total = 0
    nb_total = 0
    
    for i in range(num_pairs):

        #print("data = ", data)
        
        a = data[i][0]

        #print("a = ", a)
        b = data[i][1]
        na = data[i][2]
        nb = data[i][3]
    
        #w_ab = majority_vote_prob(a,b,params,dim,num_items)

        a = int(float(a))
        b = int(float(b))
        
        mu_a = params[0:dim,a]
        mu_b = params[0:dim,b]

        p_val = np.zeros(dim)

        for j in range(dim):
            
            p_val[j] = 1 - norm.cdf(0, mu_a[j] - mu_b[j], 1/math.sqrt(1))

        w_ab = p_val[0]*p_val[1]*p_val[2] + (1 - p_val[0])*p_val[1]*p_val[2] + p_val[0]*(1 - p_val[1])*p_val[2] + p_val[0]*p_val[1]*(1 - p_val[2])
        #w_ba = (1 - p_val[0])*(1 - p_val[1])*(1 - p_val[2]) + p_val[0]*(1 - p_val[1])*(1 - p_val[2]) + (1 - p_val[0])*p_val[1]*(1 - p_val[2]) + (1 - p_val[0])*(1 - p_val[1])*p_val[2]
        w_ba = 1 - w_ab

        epsilon = 0.01
        
        if w_ab == 0:
            w_ab = epsilon
            w_ba = 1 - w_ab
        elif w_ab == 1:
            w_ba = epsilon
            w_ab = 1 - w_ba

        #print("w_ab = ", w_ab)

        if flag == 'test':

            prob_estimated[a][b] = w_ab
            prob_estimated[b][a] = 1 - w_ab

            #prob_testdata[a][b] = ((int(float(na)))+1)/(int(float(na)) + int(float(nb)) + 2)
            prob_testdata[a][b] = (int(float(na)))/(int(float(na)) + int(float(nb)))
            prob_testdata[b][a] = 1 - prob_testdata[a][b]

            if ((prob_estimated[a][b] - 0.5)*(prob_testdata[a][b] - 0.5)) > 0:
                pred_accuracy = pred_accuracy + (int(float(na)) + int(float(nb)))

            rmse = rmse + ((prob_estimated[a][b] - prob_testdata[a][b])**2)*(int(float(na)) + int(float(nb)))

            na_total = na_total + int(float(na))
            nb_total = nb_total + int(float(nb))

            #print("prob_estimated", prob_estimated[a][b], "prob_testdata", prob_testdata[a][b])          

        
            
        ll = ll + int(float(na)) * math.log2(w_ab) + int(float(nb)) * math.log2(w_ba)

        #for j in range(dim):

        fn_der[0][a] = fn_der[0][a] + (int(float(na))/w_ab - int(float(nb))/(1 - w_ab)) * norm.pdf(mu_a[0] - mu_b[0])*(p_val[1] + p_val[2] - 2*p_val[1]*p_val[2])
        fn_der[1][a] = fn_der[1][a] + (int(float(na))/w_ab - int(float(nb))/(1 - w_ab)) * norm.pdf(mu_a[1] - mu_b[1])*(p_val[0] + p_val[2] - 2*p_val[0]*p_val[2])
        fn_der[2][a] = fn_der[2][a] + (int(float(na))/w_ab - int(float(nb))/(1 - w_ab)) * norm.pdf(mu_a[2] - mu_b[2])*(p_val[0] + p_val[1] - 2*p_val[0]*p_val[1])
        
        fn_der[0][b] = fn_der[0][b] - (int(float(na))/w_ab - int(float(nb))/(1 - w_ab)) * norm.pdf(mu_a[0] - mu_b[0])*(p_val[1] + p_val[2] - 2*p_val[1]*p_val[2])
        fn_der[1][b] = fn_der[1][b] - (int(float(na))/w_ab - int(float(nb))/(1 - w_ab)) * norm.pdf(mu_a[1] - mu_b[1])*(p_val[0] + p_val[2] - 2*p_val[0]*p_val[2])
        fn_der[2][b] = fn_der[2][b] - (int(float(na))/w_ab - int(float(nb))/(1 - w_ab)) * norm.pdf(mu_a[2] - mu_b[2])*(p_val[0] + p_val[1] - 2*p_val[0]*p_val[1])
        
    fn_der = fn_der.flatten()

    

    
    
    if flag == 'test':

        #rmse = math.sqrt(rmse/num_pairs)
        pred_accuracy = pred_accuracy/(na_total + nb_total)
        rmse = math.sqrt(rmse/(na_total + nb_total))
        #print("rmse = ", rmse)
        return (-ll), fn_der, rmse, pred_accuracy
    else:
        return (-ll), fn_der

    '''if flag == 'test':
        return (-ll), rmse, pred_accuracy
    else:
        return (-ll)'''


def majority_vote_ll_DFData(params, test_set1, test_set2, dim, num_items):

    ll = 0
    fn_der = np.zeros((dim,num_items))

    prob_testdata = np.zeros((num_items, num_items))
    
    
    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    num_pairs = test_data1.shape[0]
    
    for i in range(num_pairs):  #test pairs

        if test_set1[i,-1] == 1:

            na = 1
            nb = 0
        elif test_set1[i,-1] == -1:

            na = 0
            nb = 1
        
        ind1 = np.argmax(test_data1[i,:])
        ind2 = np.argmax(test_data2[i,:])

        a = ind1
        b = ind2

        mu_a = params[0:dim,ind1]
        mu_b = params[0:dim,ind2]

        p_val = np.zeros(dim)

        for j in range(dim):
            
            p_val[j] = 1 - norm.cdf(0, mu_a[j] - mu_b[j], 1/math.sqrt(1))

        w_ab = p_val[0]*p_val[1]*p_val[2] + (1 - p_val[0])*p_val[1]*p_val[2] + p_val[0]*(1 - p_val[1])*p_val[2] + p_val[0]*p_val[1]*(1 - p_val[2])
        #w_ba = (1 - p_val[0])*(1 - p_val[1])*(1 - p_val[2]) + p_val[0]*(1 - p_val[1])*(1 - p_val[2]) + (1 - p_val[0])*p_val[1]*(1 - p_val[2]) + (1 - p_val[0])*(1 - p_val[1])*p_val[2]
        w_ba = 1 - w_ab

        epsilon = 0.01
        
        if w_ab == 0:
            w_ab = epsilon
            w_ba = 1 - w_ab
        elif w_ab == 1:
            w_ba = epsilon
            w_ab = 1 - w_ba

        
            
        ll = ll + na * math.log2(w_ab) + nb * math.log2(w_ba)
      
        

        #for j in range(dim):

        fn_der[0][a] = fn_der[0][a] + (na/w_ab - nb/(1 - w_ab)) * norm.pdf(mu_a[0] - mu_b[0])*(p_val[1] + p_val[2] - 2*p_val[1]*p_val[2])
        fn_der[1][a] = fn_der[1][a] + (na/w_ab - nb/(1 - w_ab)) * norm.pdf(mu_a[1] - mu_b[1])*(p_val[0] + p_val[2] - 2*p_val[0]*p_val[2])
        fn_der[2][a] = fn_der[2][a] + (na/w_ab - nb/(1 - w_ab)) * norm.pdf(mu_a[2] - mu_b[2])*(p_val[0] + p_val[1] - 2*p_val[0]*p_val[1])
        
        fn_der[0][b] = fn_der[0][b] - (na/w_ab - nb/(1 - w_ab)) * norm.pdf(mu_a[0] - mu_b[0])*(p_val[1] + p_val[2] - 2*p_val[1]*p_val[2])
        fn_der[1][b] = fn_der[1][b] - (na/w_ab - nb/(1 - w_ab)) * norm.pdf(mu_a[1] - mu_b[1])*(p_val[0] + p_val[2] - 2*p_val[0]*p_val[2])
        fn_der[2][b] = fn_der[2][b] - (na/w_ab - nb/(1 - w_ab)) * norm.pdf(mu_a[2] - mu_b[2])*(p_val[0] + p_val[1] - 2*p_val[0]*p_val[1])
        
    fn_der = fn_der.flatten()
    
    print("ll = ", ll)

    return (-ll), fn_der
    

    


def get_accuracy(params, data, dim, num_items, num_pairs):
   
    data = np.reshape(data, (num_pairs,4))

    prob_data = np.zeros((num_items, num_items))
    
    accuracy = 0
    pred_accuracy = 0
    rmse = 0
    
    na_total = 0
    nb_total = 0

    for i in range(num_pairs):

        a = data[i][0]
        b = data[i][1]
        na = data[i][2]
        nb = data[i][3]
        p_a = 0
        p_b = 0

        prob_data[a][b] = (int(float(na)))/(int(float(na)) + int(float(nb)))
        prob_data[b][a] = 1 - prob_data[a][b]
    
        lg = majority_vote_prob(a,b,params,dim,num_items)

        if lg > 0.5:
            p_a += 1
        elif lg < 0.5:
            p_b += 1

        a = int(float(a))
        b = int(float(b))

        accuracy = accuracy + int(float(na)) * p_a + int(float(nb)) * p_b
        na_total = na_total + int(float(na))
        nb_total = nb_total + int(float(nb))

        rmse = rmse + ((lg - prob_data[a][b])**2)*(int(float(na)) + int(float(nb)))

        if (lg - 0.5) * (prob_data[a][b] - 0.5) > 0:            
            pred_accuracy = pred_accuracy + (int(float(na)) + int(float(nb)))


        #print("na = ", na, "nb = ", nb)

        

    acc1 = accuracy/(int(float(na_total)) + int(float(nb_total)))

    print("accuracy = ", acc1)

    pred_accuracy = pred_accuracy/(na_total + nb_total)

    print("pred_accuracy = ", pred_accuracy)

    rmse = math.sqrt(rmse/(na_total + nb_total))

    print("rmse = ", rmse)

    return acc1, pred_accuracy, rmse

def get_accuracy_synthetic(params, dim, num_items, prob_test, score_DFdata,test_set1, test_set2):
       
    accuracy = 0

    score = np.zeros(num_items)

    total_pairs = int(comb(num_items,2))

    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    num_pairs = test_data1.shape[0]
    rmse = 0
    prob_estimated = np.zeros(num_pairs)
    
    for i in range(num_pairs):  #test pairs
        
        ind1 = np.argmax(test_data1[i,:])
        ind2 = np.argmax(test_data2[i,:])
             
        lg = majority_vote_prob(ind1,ind2,params,dim,num_items)

        prob_estimated[i] = lg

        if (lg - 0.5) * (prob_test[i] - 0.5) > 0:
            accuracy += 1

        rmse = rmse + (lg - prob_test[i])**2


    acc1 = accuracy/num_pairs

    rmse = math.sqrt(rmse/num_pairs)

    print("rmse = ", rmse)

    print("accuracy = ", acc1)

    #Copeland Ranking

    for i in range(num_items):
        for j in range(i+1, num_items):

            lg = majority_vote_prob(i,j,params,dim,num_items)
            
            if lg > 0.5:
                score[i] += 1
            elif lg < 0.5:
                score[j] += 1


    print("score = ", score)

    #np.savetxt("prob_MV.txt", prob_estimated)
    #np.savetxt("prob_test.txt", prob_test)

    ktc2, p_val = stats.kendalltau(score_DFdata,score)
    print(ktc2)

    upsets = 0
    
    for i in range(num_pairs):  #test pairs

        ind1 = np.argmax(test_data1[i,:])
        ind2 = np.argmax(test_data2[i,:])
             
        lg = majority_vote_prob(ind1,ind2,params,dim,num_items)

        if (score[ind1] - score[ind2]) * (prob_test[i] - 0.5) < 0:
            upsets += 1

    upsets = upsets/num_pairs

    print("upsets = ", upsets)


    return acc1, ktc2, upsets, rmse


def obj_fn(params, data):

    #num_items = 100
    #dim = 10
 
    dim = data[-1]
    num_pairs = data[-2]
    num_items = data[-3]

    num_pairs = int(float(num_pairs))
    num_items = int(float(num_items))
    dim = int(float(dim))

    #print(num_pairs)
    data2 = data[0:(num_pairs*4)]

    params = np.reshape(params, (dim,num_items))

    
    for i in range(num_items):
        
        mu = params[0:dim,i]
        
    fn, fn_der = majority_vote_ll(params, data2, dim, num_items,num_pairs, 'train') #bfgs

    #fn = majority_vote_ll(params, data2, dim, num_items,num_pairs, 'train') #nelder-mead

    #fn = majority_vote_ll(params, data2, dim, num_items,num_pairs)
    
    return fn, fn_der
       
 

def MV_MLE_DFdata(embedding_obj, params, num_items, l, m, dim, seed, score_DF):

    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)
    #validation_data1 = np.copy(embedding_obj.validation_data1)
    #validation_data2 = np.copy(embedding_obj.validation_data2)

    #score_DFdata = np.copy(embedding_obj.score)

    score_DFdata = np.copy(score_DF)

    prob_test = np.copy(embedding_obj.prob_test)
    rank_data12 = np.copy(embedding_obj.rank_data12)
    rank_data22 = np.copy(embedding_obj.rank_data22)

    items_ranked_original = np.copy(embedding_obj.items_ranked_original)

    total_pairs = int(comb(num_items,2))
    #num_pairs = m

    #train_set1[train_set1 == -1] = 0

    print("m = ", m)
   
    
    data, num_pairs = create_data.create_data(train_set1, train_set2, num_items, 'train', seed)
    

    #flatten the nd arrays and pass as x0 or args in minimize
    print(data.shape[0],data.shape[1])
    data = data.flatten()
    print(data.shape[0])
    params = params.flatten()

    print(num_pairs)

    data2 = np.append(data, num_items)
    data2 = np.append(data2, num_pairs)
    data2 = np.append(data2, dim)
    
    print("Hello")

    #f, gradf = obj_fn(params, data2)

    n_iter = 100
    learning_rate = 0.01
    
    params_gd = grad_descent(params, data2, n_iter, learning_rate, dim, num_items, num_pairs)
    weights_bc = params_gd
    
    #res = minimize(obj_fn, x0 = params, args = data2, method = 'BFGS', jac = True, tol = 1e-6, options = {'maxiter': 10000}) #minimize the negative log likelihood
    #weights_bc = res.x
    weights_bc = np.reshape(weights_bc, (dim,num_items))

    #valid_data, num_pairs_val = create_data.create_data(validation_data1, validation_data2, num_items, 'train', seed)

    #validation_ll, validation_ll_der = majority_vote_ll_DFData(weights_bc, validation_data1, validation_data2, dim, num_items)

    #print("validation_ll = ", validation_ll)

    #test_data, num_pairs_test = create_data.create_data(test_set1, test_set2, num_items, 'test', seed)
    
    test_ll, test_ll_der = majority_vote_ll_DFData(weights_bc, test_set1, test_set2, dim, num_items)
    test_acc, ktc, upsets, rmse = get_accuracy_synthetic(weights_bc, dim, num_items, prob_test, score_DFdata, test_set1, test_set2)


    print(test_acc)
    #print(weights_bc)
    
    print(test_ll)
        
    return test_acc, ktc, upsets, rmse
    #return pred_accuracy, ktc

#def MV_MLE_Realdata(num_items, params, dim_mv, seed):
#def MV_MLE_Realdata(train_data1, train_data2, test_data1, test_data2, num_items, prob_data, params, dim_mv, seed):
def MV_MLE_Realdata(train_data, valid_data, test_data, num_items, params, dim_mv, seed, data_name):
    

    total_pairs = int(comb(num_items,2))


    dim = dim_mv

    #data = np.loadtxt("data_majority_vote_train.txt", dtype = float)
    #data = np.loadtxt("jester_train"+str(seed)+".txt", dtype = float)
    #data = np.loadtxt("movielenstrain"+str(seed)+".txt", dtype = float)

    #data, num_pairs = create_data.create_data(train_data1, train_data2, num_items)

    #data, train_pairs = create_data3.create_data(train_data, num_items, 'train', seed, data_name)
    #valid_data, valid_pairs = create_data3.create_data(valid_data, num_items, 'validation', seed, data_name)
    #test_data, test_pairs = create_data3.create_data(test_data, num_items, 'test', seed, data_name)

    data = np.loadtxt(data_name+"_modifiedtrain"+str(seed)+".txt", dtype = float)
    valid_data = np.loadtxt(data_name+"_modifiedvalidation"+str(seed)+".txt", dtype = float)
    test_data = np.loadtxt(data_name+"_modifiedtest"+str(seed)+".txt", dtype = float)
    
    data = data.astype(int)
    print(data)

    num_pairs = data.shape[0]
    train_pairs = data.shape[0]
    valid_pairs = valid_data.shape[0]
    test_pairs = test_data.shape[0]
    print("num_pairs = ", num_pairs, "train_pairs = ", train_pairs)
    
    

    #flatten the nd arrays and pass as x0 or args in minimize
    print(data.shape[0],data.shape[1])
    data = data.flatten()
    print(data.shape[0])
    params = params.flatten()

    #print(num_pairs)

    data2 = np.append(data, num_items)
    data2 = np.append(data2, num_pairs)
    data2 = np.append(data2, dim)
    
    print("Hello")

    #f, gradf = obj_fn(params, data2)

    n_iter = 100
    print("iter = ", n_iter)
    #learning_rate = 0.005 #MV
    learning_rate = 0.005 #HotS 
    #learning_rate = 0.005 #WoL
    #learning_rate = 0.0005 #Dota
    #learning_rate = 0.005 #Dota2
    #learning_rate = 0.0001 #0.00005 #jester, n_iter = 20
        
    params_gd = grad_descent(params, data2, n_iter, learning_rate, dim, num_items, num_pairs)
    weights_bc = params_gd
    
    #res = minimize(obj_fn, x0 = params, args = data2, method = 'BFGS', jac = True, tol = 1e-6, options = {'maxiter': 10000, 'disp': True, 'return_all': True}) #minimize the negative log likelihood
    #res = minimize(obj_fn, x0 = params, args = data2, method = 'Nelder-Mead', tol = 1e-4, options = {'maxiter': 1000})
    #weights_bc = res.x
    weights_bc = np.reshape(weights_bc, (dim,num_items))

    

    validation_ll, validation_ll_der = majority_vote_ll(weights_bc, valid_data,dim, num_items, valid_pairs, 'validation')

    print("validation_ll = ", validation_ll)

    

    #test_data = np.loadtxt("data_majority_vote_test.txt", dtype = float)
    #test_data = np.loadtxt("jester_test"+str(seed)+".txt", dtype = float)
    #test_data = np.loadtxt("movielenstest"+str(seed)+".txt", dtype = float)
    
    test_data = test_data.astype(int)
    print(test_data)

    num_pairs_test = test_data.shape[0]
    
    test_ll, test_ll_der, rmse, pred_accuracy = majority_vote_ll(weights_bc, test_data,dim, num_items, test_pairs, 'test')
    #test_ll, rmse, pred_accuracy = majority_vote_ll(weights_bc, test_data,dim, num_items, num_pairs_test, 'test')
    test_acc, pred_accuracy1, rmse2 = get_accuracy(weights_bc, test_data,dim, num_items, num_pairs_test)

    '''data = data.reshape(num_pairs,4)

    for i in range(num_pairs):

        a = data[i,0]
        b = data[i,1]
        na = data[i,2]
        nb = data[i,3]
    
        lg = logistic(matchup(a,b))'''

    print(test_acc)
    print(test_ll)
    print(rmse2)
    print(pred_accuracy)
    print(pred_accuracy1)
    
    return test_ll, pred_accuracy, rmse2
    #return pred_accuracy, ktc

