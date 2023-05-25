import numpy as np
from numpy import random
import math
from scipy import stats
#import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

import pickle
from scipy.special import comb
import pandas as pd
import os
import sys
import pairwise_comparisonsC
import gen_embedding
import gen_embedding_rank
import RC
#import SFmodel_MLE
import SFmodel_Copeland
import DFlearn
import LRPR
import majority_vote3
import blade_chest_Synthetic_DFData
import BC_Accuracy
import GNNRank_Synthetic
#import Tournament_generation_DFmodel


#idx = sys.argv[1]

num_items = 100
dim = 3

total_pairs = int(comb(num_items,2))

m = int(num_items * math.log2(num_items))

l2 = int(math.log2(num_items))


#r = int(total_pairs/m2)

m_vals = []
l_vals = []
#no_of_pairs = np.zeros(r-1)
no_of_comparisons = np.zeros(6)

'''
for i in range(1,r):
    m3 = i*m2
    m_vals.append(m3)
    #no_of_pairs[i-1] = m3

no_of_pairs = m_vals


print(m_vals)
print(no_of_pairs)
'''
r = 7

for i in range(1,7,1):
    l3 = i*l2
    l_vals.append(l3)

no_of_comparisons = l_vals
print(no_of_comparisons)

accuracy_no_of_pairs = np.zeros(((10,r-1)))
ktc_vals = np.zeros(((10,r-1)))
ktc2_vals = np.zeros(((10,r-1)))


dim2 = 10
c = 0.00009 #l2 regularization parameter
#h_val = h_1
#num_clusters = 1

'''accuracy_RC_total = []
ktc_RC_total = []

accuracy_DFlearn_total = []
ktc_DFlearn_total = []

accuracy_SF_total = []
ktc_SF_total = []'''

accuracy_RC_total = np.zeros(((10,r-1)))
accuracy_DFlearn_total = np.zeros(((10,r-1)))
accuracy_SF_total = np.zeros(((10,r-1)))
accuracy_LRPR_total = np.zeros(((10,r-1)))
accuracy_MV_total = np.zeros(((10,r-1)))
accuracy_BC_total = np.zeros(((10,r-1)))


cycles_total = np.zeros(((10,r-1)))
cycle_fraction_total = np.zeros(((10,r-1)))

ktc_RC_total = np.zeros(((10,r-1)))
ktc_DFlearn_total = np.zeros(((10,r-1)))
ktc_SF_total = np.zeros(((10,r-1)))
ktc_LRPR_total = np.zeros(((10,r-1)))
ktc_MV_total = np.zeros(((10,r-1)))
ktc_BC_total = np.zeros(((10,r-1)))

        
upset_RC_total = np.zeros(((10,r-1)))
upset_SF_total = np.zeros(((10,r-1)))
upset_DFlearn_total = np.zeros(((10,r-1)))
upset_LRPR_total = np.zeros(((10,r-1)))
upset_MV_total = np.zeros(((10,r-1)))
upset_BC_total = np.zeros(((10,r-1)))

rmse_RC_total = np.zeros(((10,r-1)))
rmse_DFlearn_total = np.zeros(((10,r-1)))
rmse_MV_total = np.zeros(((10,r-1)))
rmse_SF_total = np.zeros(((10,r-1)))
rmse_LRPR_total = np.zeros(((10,r-1)))
rmse_BC_total = np.zeros(((10,r-1)))


dim_mv = 3
dim_BC = 10

count_iter = 0

def initialize_params_normal(num_items, dim_mv, std_deviation,t):

    np.random.seed(t)
        
    params = np.random.normal(0, scale = std_deviation, size = (dim_mv,num_items))

    params = params/np.linalg.norm(params,2,axis = 0)
       
    return params

def initialize_params_uniform(num_items, dim_mv, t):

    np.random.seed(t)
    
    params = np.random.uniform(low = -10, high = 10, size = (dim_mv,num_items))

    params = params/np.linalg.norm(params,2,axis = 0)

    return params

#val_loop = [0, 1, 2, 3 ,4 ,5 ,6 ,7 ,8 ,9]

val_loop_RC = [0,3]

for t in range(2):#val_loop_RC
    
    print(t)

    #random.seed(t)
    
    embeddings = gen_embedding.generate_embedding(num_items, dim, np.sqrt(1/1),t)
    #embeddings2 = gen_embedding.generate_embedding_new(num_items, dim, 0.001, t)
    embeddings2 = gen_embedding_rank.generate_embedding_new(num_items, dim, 0.001, t)
    #print("embeddings2 = ",embeddings2.shape)
    np.savetxt("embeddings_DF_new"+str(t)+".txt", embeddings2)
    embeddings_new = np.transpose(embeddings2)
    #print("embeddings_new = ",embeddings_new.shape)
    #print(embeddings_new)
    
    
    '''
    clustering = SpectralClustering(n_clusters=3, random_state=0).fit(embeddings_new)
    #clustering.labels_
    labels = clustering.fit_predict(embeddings_new)
    np.savetxt("labels original data"+str(t)+".txt", labels)
    print("labels = ", labels)
    #fig = plt.figure(figsize = (10, 7))
    #ax = plt.axes(projection ="3d")
    #ax.scatter3D(embeddings_new[:, 0], embeddings_new[:, 1], embeddings_new[:, 2], s = 10, linewidth=0, c=labels, cmap='flag')
    #plt.show()
    '''
    
    embeddings2 = np.loadtxt("embeddings_DF_new"+str(t)+".txt", dtype = float)
    
    
    U = np.copy(embeddings2)

    prob_data = np.zeros((num_items,(num_items+1)))

    for i in range(num_items-1):
        for j in range(i+1, num_items):

            diff = np.abs(U[:,i] - U[:,j])
            diff2 = np.amax(diff)
            ind = np.argmax(diff)
            diff3 = (U[ind,i] - U[ind,j])
            prob = 1/(1+np.exp(-diff3))
            
            prob_data[i][j] = prob
            prob_data[j][i] = 1 - prob
               

            if prob > 0.5:
                   
                prob_data[i][num_items] = prob_data[i][num_items] + 1
                   
            elif prob < 0.5:
               
                 prob_data[j][num_items] = prob_data[j][num_items] + 1

    score = np.zeros(num_items)
    keys = np.zeros(num_items)
    
    for i in range(num_items):
        score[i] = prob_data[i][num_items]
        keys[i] = i  

    #params = initialize_params_normal(num_items, dim_mv, 1/math.sqrt(1), t)

    #np.savetxt("embeddings"+str(t)+".txt", embeddings)

    #embeddings = Tournament_generation_DFmodel.generate_tournament(num_items, dim, 8, np.sqrt(1/dim))
    #embeddings = gen_embedding.generate_embedding_uniform(num_items, dim, t)
    #embeddings = gen_embedding.generate_clustered_embedding(num_items, dim, num_clusters)

    
            
    count_m = 0

    #m_vals = [664]

    
    #for m in m_vals:
    for l in l_vals:

        print("l = ", l)
    
        embedding_obj = pairwise_comparisonsC.embedding(embeddings2, num_items, l, m, t)

        
        
        #accuracy_RC, ktc_RC, upset_RC, rmse_RC = RC.RankCentrality(embedding_obj, num_items, l, m, score)
        #accuracy_LRPR, ktc_LRPR, upset_LRPR, rmse_LRPR = LRPR.LRPR(embedding_obj, num_items, l , m, score)
        #accuracy_SF, ktc_SF, upset_SF, rmse_SF = SFmodel_Copeland.SF_Copeland_DFdata(embedding_obj, num_items, l, m, dim, score)
        #accuracy_DFlearn, ktc_DFlearn, upset_DFlearn, cycles, cycle_fraction, rmse_DFlearn, learned_embeddings = DFlearn.Ranking(embedding_obj, num_items, l, m, dim, dim2, prob_data, score)
        #np.savetxt("embeddings_learned"+str(t)+"_"+str(m)+".txt", learned_embeddings)
        
        '''
        clustering = SpectralClustering(n_clusters=3, random_state=0).fit(learned_embeddings)
        #clustering.labels_
        labels = clustering.fit_predict(learned_embeddings)
        np.savetxt("labels learned data"+str(t)+"_"+str(m)+".txt", labels)
        print("labels = ", labels)
        fig = plt.figure(figsize = (10, 7))
        #ax = plt.axes(projection ="3d")
        #ax.scatter3D(learned_embeddings[:, 0], learned_embeddings[:, 1], learned_embeddings[:, 2], s = 10, linewidth=0, c=labels, cmap='flag')
        #plt.show()
        '''
        
        #params = initialize_params_normal(num_items, dim_mv, 1/math.sqrt(0.001), t)
        #params = initialize_params_uniform(num_items, dim_mv, t)
        #accuracy_MV, ktc_MV, upset_MV, rmse_MV = majority_vote3.MV_MLE_DFdata(embedding_obj, params, num_items, l, m, dim_mv, t, score)
        
        '''
        accuracy_SF, ktc_SF, rmse_SF = SFmodel_Copeland.SF_Copeland_DFdata(embedding_obj, num_items, l, m, dim)
        #accuracy_DFlearn, ktc_DFlearn, upset_DFlearn, cycles, cycle_fraction, rmse_DFlearn = DFlearn.Ranking(embedding_obj, num_items, l, m, dim, dim2) #, idx)
        accuracy_LRPR, ktc_LRPR, rmse_LRPR = LRPR.LRPR(embedding_obj, num_items, l , m)

        #params = initialize_params_normal(num_items, dim_mv, 1/math.sqrt(1), t)
        #accuracy_MV, ktc_MV, upsets_MV, rmse_MV = majority_vote3.MV_MLE_DFdata(embedding_obj, params, num_items, l, m, dim_mv, t)
        data_BC = blade_chest_Synthetic_DFData.create_data(embedding_obj, num_items, l, m, t)'''
        #accuracy_LRPR, ktc_LRPR, rmse_LRPR = LRPR.LRPR(embedding_obj, num_items, l , m)
        #accuracy_DFlearn, ktc_DFlearn, upset_DFlearn, cycles, cycle_fraction, rmse_DFlearn = DFlearn.Ranking(embedding_obj, num_items, l, m, dim, dim2)

        #data_BC = blade_chest_Synthetic_DFData.create_data(embedding_obj, num_items, l, m, t)
        #score_DFdata = np.copy(embedding_obj.score)
        
        pred_accuracy_BC, rmse_BC, ktc_BC, upset_BC = BC_Accuracy.BC_Synthetic(l, t, score)

        #GNNRank_Synthetic.DFdata(embedding_obj, num_items, l, m, t) #GNN format data generation

        accuracy_BC_total[count_iter][count_m] = pred_accuracy_BC
        ktc_BC_total[count_iter][count_m] = ktc_BC
        rmse_BC_total[count_iter][count_m] = rmse_BC
        

        '''
        accuracy_MV_total[count_iter][count_m] = accuracy_MV
        rmse_MV_total[count_iter][count_m] = rmse_MV
        ktc_MV_total[count_iter][count_m] = ktc_MV
        '''

        '''
        cycles_total[count_iter][count_m] = cycles
        cycle_fraction_total[count_iter][count_m] = cycle_fraction
        
        accuracy_RC_total[count_iter][count_m] = accuracy_RC
        accuracy_SF_total[count_iter][count_m] = accuracy_SF
        accuracy_DFlearn_total[count_iter][count_m] = accuracy_DFlearn
        accuracy_LRPR_total[count_iter][count_m] = accuracy_LRPR
        accuracy_MV_total[count_iter][count_m] = accuracy_MV
        #accuracy_BC_total[count_iter][count_m] = pred_accuracy_BC
       
        ktc_RC_total[count_iter][count_m] = ktc_RC
        ktc_SF_total[count_iter][count_m] = ktc_SF
        ktc_DFlearn_total[count_iter][count_m] = ktc_DFlearn
        ktc_LRPR_total[count_iter][count_m] = ktc_LRPR
        ktc_MV_total[count_iter][count_m] = ktc_MV
        #ktc_BC_total[count_iter][count_m] = ktc_BC

        upset_RC_total[count_iter][count_m] = upset_RC
        upset_SF_total[count_iter][count_m] = upset_SF
        upset_DFlearn_total[count_iter][count_m] = upset_DFlearn
        upset_LRPR_total[count_iter][count_m] = upset_LRPR
        upset_MV_total[count_iter][count_m] = upset_MV
        #upset_BC_total[count_iter][count_m] = upset_BC
       
        rmse_RC_total[count_iter][count_m] = rmse_RC
        rmse_DFlearn_total[count_iter][count_m] = rmse_DFlearn
        rmse_MV_total[count_iter][count_m] = rmse_MV
        rmse_SF_total[count_iter][count_m] = rmse_SF
        rmse_LRPR_total[count_iter][count_m] = rmse_LRPR
        #rmse_BC_total[count_iter][count_m] = rmse_BC
        '''
        
        '''
        accuracy_BC_total[count_iter][count_m] = pred_accuracy_BC
        ktc_BC_total[count_iter][count_m] = ktc_BC
        upset_BC_total[count_iter][count_m] = upset_BC
        rmse_BC_total[count_iter][count_m] = rmse_BC
        '''
        count_m = count_m + 1
    
    count_iter = count_iter + 1

#accuracy_reshape = accuracy_no_of_pairs.reshape(accuracy_no_of_pairs.shape[0], -1)

#ktc_vals_reshape = ktc_vals.reshape(ktc_vals.shape[0], -1)

np.savetxt("accuracy_BC_DFData.txt", accuracy_BC_total)
np.savetxt("ktc_vals_BC_DFData.txt",ktc_BC_total)
#np.savetxt("upset_BC_DFData.txt", upset_BC_total)
np.savetxt("rmse_BC_DFData.txt", rmse_BC_total)

'''
np.savetxt("accuracy_RC_DFData.txt", accuracy_RC_total)
np.savetxt("accuracy_SF_DFData.txt", accuracy_SF_total)
np.savetxt("accuracy_DF_DFData.txt", accuracy_DFlearn_total)

np.savetxt("accuracy_MV_DFData.txt", accuracy_MV_total)
#np.savetxt("accuracy_BC_DFData.txt", accuracy_BC_total)

np.savetxt("cycles.txt", cycles_total)
np.savetxt("cycle_fraction.txt", cycle_fraction_total)


np.savetxt("ktc_vals_RC_DFData.txt", ktc_RC_total)
np.savetxt("ktc_vals_SF_DFData.txt", ktc_SF_total)
np.savetxt("ktc_vals_DF_DFData.txt", ktc_DFlearn_total)
np.savetxt("ktc_vals_LRPR_DFData.txt", ktc_LRPR_total)
np.savetxt("ktc_vals_MV_DFData.txt", ktc_MV_total)
#np.savetxt("ktc_vals_BC_DFData.txt",ktc_BC_total)

np.savetxt("upset_RC_DFData.txt", upset_RC_total)
np.savetxt("upset_SF_DFData.txt", upset_SF_total)
np.savetxt("upset _DF_DFData.txt", upset_DFlearn_total)
np.savetxt("upset_LRPR_DFData.txt", upset_LRPR_total)
np.savetxt("upset_MV_DFData.txt", upset_MV_total)
#np.savetxt("upset_BC_DFData.txt", upset_BC_total)

np.savetxt("rmse_RC_DFData.txt", rmse_RC_total)
np.savetxt("rmse_DFlearn_DFData.txt", rmse_DFlearn_total)
np.savetxt("rmse_MV_DFData.txt", rmse_MV_total)
np.savetxt("rmse_SF_DFData.txt", rmse_SF_total)
np.savetxt("rmse_LRPR_DFData.txt", rmse_LRPR_total)
#np.savetxt("rmse_BC_DFData.txt", rmse_BC_total)
'''

'''acc_RC = np.loadtxt("accuracy_RC_DFData.txt")
acc_DF = np.loadtxt("accuracy_DF_DFData.txt")
acc_SF = np.loadtxt("accuracy_SF_DFData.txt")

acc_RC_mean = np.mean(acc_RC, axis = 0)
acc_SF_mean = np.mean(acc_SF, axis = 0)
acc_DF_mean = np.mean(acc_DF, axis = 0)

acc_RC_err = stats.sem(acc_RC, axis = 0)
acc_SF_err = stats.sem(acc_SF, axis = 0)
acc_DF_err = stats.sem(acc_DF, axis = 0)

plt.errorbar(no_of_pairs, acc_RC_mean, yerr = acc_RC_err)
plt.errorbar(no_of_pairs, acc_SF_mean, yerr = acc_SF_err)
plt.errorbar(no_of_pairs, acc_DF_mean, yerr = acc_DF_err)
plt.legend(["RankCentrality", "SF_MLE", "DFLearn"])#, loc = "upper left")
plt.title("DF Model Data", size = 20)
plt.xlabel("No. of training pairs", size = 20)
plt.ylabel("Pairwise Prediction Accuracy", size = 20)
plt.show()'''
