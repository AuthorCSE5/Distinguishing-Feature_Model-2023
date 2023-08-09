import numpy as np
import pandas as pd
import re
import math
from scipy.special import comb
from matplotlib import pyplot as plt
#from statsmodels import api
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


import create_data
import create_data3

from sklearn.model_selection import GridSearchCV



  

def matchup(a,b,params,dim,num_items,choice):

    a = int(float(a))
    b = int(float(b))
    
    a_blade = params[0:dim,a]
    a_chest = params[dim:2*dim,a]
    b_blade = params[0:dim,b]
    b_chest = params[dim:2*dim,b]
    gamma_a = params[2*dim,a]
    gamma_b = params[2*dim,b]
    
    if choice == 'inner':
        return np.dot(a_blade, b_chest) - np.dot(b_blade, a_chest) + gamma_a - gamma_b

    if choice == 'dist':
        return (np.linalg.norm(b_blade - a_chest))**2 - (np.linalg.norm(a_blade - b_chest))**2 + gamma_a - gamma_b

def logistic(x):

    return 1/(1+math.exp(-x))

def get_accuracy(params, data, dim, num_items, num_pairs, choice):
   
    accuracy = 0
    pred_accuracy = 0
    rmse = 0
    upsets = 0

    prob_data = np.zeros((num_items, num_items))
    prob_estimated = np.zeros((num_items, num_items))

    
    na_total = 0
    nb_total = 0

    print("data = ", data)

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
    
        lg = logistic(matchup(a,b,params,dim,num_items,choice))

        if lg > 0.5:
            p_a += 1
        elif lg < 0.5:
            p_b += 1

        #print("na = ", na, "nb = ", nb)

        prob_estimated[a][b] = lg
        prob_estimated[b][a] = 1 - lg
        
        if ((prob_data[a][b] - 0.5)*(prob_estimated[a][b] - 0.5)) > 0:
            pred_accuracy = pred_accuracy + (int(float(na)) + int(float(nb)))
            
       

        rmse = rmse + ((prob_estimated[a][b] - prob_data[a][b])**2)*(int(float(na)) + int(float(nb)))

        

        accuracy = accuracy + int(float(na)) * p_a + int(float(nb)) * p_b
        na_total = na_total + int(float(na))
        nb_total = nb_total + int(float(nb))

    acc1 = accuracy/(int(float(na_total)) + int(float(nb_total)))
    
    
    pred_accuracy = pred_accuracy/(na_total + nb_total)

    print("pred_accuracy = ", pred_accuracy)

    rmse = math.sqrt(rmse/(na_total + nb_total))

    print("rmse = ", rmse)
    

    return acc1, pred_accuracy, rmse

def get_accuracy_Synthetic(params, data, prob_test, dim, num_items, num_pairs, choice, score_orig):
   
    accuracy = 0
    pred_accuracy = 0
    rmse = 0
    upsets = 0

    

    score_DF = np.copy(score_orig)

    prob_data = np.zeros((num_items, num_items))
    prob_estimated = np.zeros((num_items, num_items))

    
    na_total = 0
    nb_total = 0

    print("data = ", data)

    

    score = np.zeros(num_items)

    

    #Copeland Ranking
    
    for i in range(num_items-1):
        for j in range(i+1, num_items):

            lg = logistic(matchup(i,j,params,dim,num_items,choice))
            
            if lg > 0.5:
                score[i] += 1
            elif lg < 0.5:
                score[j] += 1

    print("score = ", score) 

    

    for i in range(num_pairs):

        a = data[i][0]
        b = data[i][1]
        na = data[i][2]
        nb = data[i][3]
        p_a = 0
        p_b = 0
        

        a = int(float(a))
        b = int(float(b))
    
    
        lg = logistic(matchup(a,b,params,dim,num_items,choice))

    
        prob_estimated[a][b] = lg
        prob_estimated[b][a] = 1 - lg
                
        if ((prob_test[i] - 0.5)*(prob_estimated[a][b] - 0.5)) > 0:
            pred_accuracy = pred_accuracy + 1

        

        rmse = rmse + ((prob_estimated[a][b] - prob_test[i])**2)

        

        if (score[a] - score[b]) * (prob_test[i] - 0.5) < 0:
            upsets += 1

          
    pred_accuracy = pred_accuracy/num_pairs

    print("pred_accuracy = ", pred_accuracy)

    rmse = math.sqrt(rmse/num_pairs)

    print("rmse = ", rmse)

    upsets = upsets/num_pairs

    print("upsets = ", upsets)

    ktc, p_val = stats.kendalltau(score,score_DF)
    print("ktc = ",ktc)

        

    return pred_accuracy, rmse, ktc





def BC(data_name, num_items, seed):

    
   
    dim = 50 
    choice = 'inner'
    

    data = np.array(pd.read_csv(data_name+'.csv'))       ##Change the name of the dataset in place of WoL

    

    print(data)

    #num_items = 4381  ##change the number of items a/c to the dataset
    

    test_indices = np.loadtxt("test_indices_BC_"+data_name+"_"+str(seed)+".txt", dtype = int) ##Change the name of the dataset in place of WoL (this text file is saved from the blade chest software to know the dataset indices used for test data)
    
    data = data[num_items+1:,0:]
    test = (data[:,0] == 'FOR_TESTING')
    test0 = np.where(test)
    test1 = test_indices

    print("test_data0 = ", test0)
    print("test_data1 = ", test1)
    test2 = data[test1]
    test3 = data[test0]
    print("test_data2 = ", test2)
    print("test_data3 = ", test3)
    test_data = test2[:,0:]
    #test_data = test3[:,1:]
    print("test_data = ", test_data)
    test_data = np.asarray(test_data, dtype = 'int')
    print("test_data = ", test_data)

    test_data_test = test_data[:,0:2]
    print("test_data_test = ", test_data_test)
    test_data_test = np.unique(test_data_test, axis = 0)
    print("test_data_test = ", test_data_test, "size = ", test_data_test.shape[0])

  
    num_pairs_test = test_data.shape[0]

    print("num_pairs = ", num_pairs_test)

    chest_vecs = np.array(pd.read_csv("BC_"+data_name+"_"+str(seed)+".txt", header = None, skiprows = 5, nrows = num_items, delimiter = ' '))     ##Change the name of the dataset in place of WoL
    blade_vecs = np.array(pd.read_csv("BC_"+data_name+"_"+str(seed)+".txt", header = None, skiprows = 5+num_items+1, nrows = num_items, delimiter = ' '))

    params_bc_ranks = np.array(pd.read_csv("BC_"+data_name+"_"+str(seed)+".txt", header = None, skiprows = 5+(2*num_items)+2, nrows = 1, delimiter = ' '))


    
    #chest_vecs = np.transpose(chest_vecs)
    #blade_vecs = np.transpose(blade_vecs)
    #params_bc_updated = np.append(blade_vecs, chest_vecs, axis = 0)
    
    params_bc_updated = np.append(blade_vecs, chest_vecs, axis = 1)
    params_bc_updated = np.transpose(params_bc_updated)

    

    #params_bc_ranks = np.transpose(params_bc_ranks)
    params_bc_ranks = np.reshape(params_bc_ranks, (1,num_items))
    params_bc_updated = np.append(params_bc_updated, params_bc_ranks, axis = 0)
        
    test_data, test_pairs = create_data3.create_data(test_data, num_items, 'test', seed, data_name)

    
    test_acc, pred_accuracy, rmse = get_accuracy(params_bc_updated, test_data, dim, num_items, test_pairs,choice)

    #test_acc, pred_accuracy, rmse = get_accuracy(params_bc_updated, test_data, dim, num_items, num_pairs_test,choice)

    
    print("test_acc = ",test_acc )
    
    print(pred_accuracy)
    
    return test_acc, pred_accuracy, rmse
    

def BC_Synthetic(m, seed, score_orig, data_name):

    prob_test = np.loadtxt("BC_"+data_name+"Data/BC_prob_test_"+data_name+"Data_"+str(m)+"_"+str(seed)+".txt", dtype = float)
    
    print(prob_test)
   
    dim = 50 
    choice = 'inner'
    num_items = 100
    
    df = pd.read_csv('BC_"+data_name+"Data/BC_"+data_name+"data_train'+str(m)+'_'+str(seed)+'.txt', sep=r'\s*(:\n\n)\s*', skiprows=0, engine='python',header=None)
    
    df.columns  = ['column1']
   

    df.drop(np.arange(0,(num_items + 2)),axis=0,inplace=True,index=None)
    df.reset_index(drop=True)

    df[["column1","column2","column3"]] = df['column1'].apply(lambda x: pd.Series(str(x).split(" ")))

    temp2 = np.array(df['column2'].apply(lambda x: pd.Series(str(x).split(":"))))
    temp3 = np.array(df['column3'].apply(lambda x: pd.Series(str(x).split(":"))))

    df["column2"] = temp2[:,0]
    df["column3"] = temp2[:,1]
    df["column4"] = temp3[:,0]
    df["column5"] = temp3[:,1]

    df.reset_index(drop=True)

    data = np.array(df)
    
    print(data)

    #data = data[num_items+1:,0:]
    test = (data[:,0] == 'FOR_TESTING')
    test0 = np.where(test)
    

    print("test_data0 = ", test0)
    
    
    test3 = data[test0]
    
    print("test_data3 = ", test3)
    
    test_data = test3[:,1:]
    
    print("test_data = ", test_data)
    
    test_data = np.asarray(test_data, dtype = 'int')
    
    print("test_data = ", test_data)

   
    
    num_pairs_test = test_data.shape[0]

    print("num_pairs = ", num_pairs_test)

    chest_vecs = np.array(pd.read_csv("BC_"+data_name+"Data/"+str(m)+"_"+str(seed)+".txt", header = None, skiprows = 5, nrows = num_items, delimiter = ' '))
    blade_vecs = np.array(pd.read_csv("BC_"+data_name+"Data/"+str(m)+"_"+str(seed)+".txt", header = None, skiprows = 5+num_items+1, nrows = num_items, delimiter = ' '))

    params_bc_ranks = np.array(pd.read_csv("BC_"+data_name+"Data/"+str(m)+"_"+str(seed)+".txt", header = None, skiprows = 5+(2*num_items)+2, nrows = 1, delimiter = ' '))

    params_bc_updated = np.append(blade_vecs, chest_vecs, axis = 1)
    params_bc_updated = np.transpose(params_bc_updated)

    params_bc_ranks = np.reshape(params_bc_ranks, (1,num_items))
    params_bc_updated = np.append(params_bc_updated, params_bc_ranks, axis = 0)
        
    pred_accuracy, rmse, ktc = get_accuracy_Synthetic(params_bc_updated, test_data, prob_test, dim, num_items, num_pairs_test,choice, score_orig)

    
    print("pred_accuracy = ", pred_accuracy)
    
    return pred_accuracy, rmse, ktc
    

