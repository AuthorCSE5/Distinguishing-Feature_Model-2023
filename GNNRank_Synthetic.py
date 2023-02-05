import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
from scipy.special import comb
import pandas as pd
from sklearn.model_selection import train_test_split

import create_data3
import GNNRank_Accuracy


def DFdata(embedding_obj, num_items, l, m, seed):

    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)
    #validation_data1 = np.copy(embedding_obj.validation_data1)
    #validation_data2 = np.copy(embedding_obj.validation_data2)


    prob_test = np.copy(embedding_obj.prob_test)
    rank_data12 = np.copy(embedding_obj.rank_data12)
    rank_data22 = np.copy(embedding_obj.rank_data22)

    items_ranked_original = np.copy(embedding_obj.items_ranked_original)

    total_pairs = int(comb(num_items,2))
    
    
    data, num_pairs = create_data3.create_data_synthetic(train_set1, train_set2, num_items, seed, m)

    ##below is for SFData/BTLData

    
    #np.savetxt("GNNRank_Datasets_finer/Synthetic/SF/"+str(m)+"_"+str(seed)+"_testdata1_SF.txt", test_set1)
    #np.savetxt("GNNRank_Datasets_finer/Synthetic/SF/"+str(m)+"_"+str(seed)+"_testdata2_SF.txt", test_set2)

    ##below is for DFData

    np.savetxt("GNNRank_Datasets_finer/Synthetic/"+str(m)+"_"+str(seed)+"_testdata1.txt", test_set1)
    np.savetxt("GNNRank_Datasets_finer/Synthetic/"+str(m)+"_"+str(seed)+"_testdata2.txt", test_set2)
    
