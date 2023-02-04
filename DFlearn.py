from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import pickle
from scipy.special import comb
import pandas as pd

from scipy import stats
from tensorflow import keras
from tensorflow.keras import backend as K
#import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Dense, Input, Lambda, Dropout, Maximum, Dot
from tensorflow.keras.models import Model
import pairwise_comparisonsC
import gen_embedding



def f(h_uv):
    
    hu = h_uv[0]
    hv = h_uv[1]
    
    
    w_s = tf.subtract(tf.cast(hu,tf.float64),tf.cast(hv,tf.float64)) #required for the dot product
    w = tf.abs(w_s) #to calculate index

    w_max = tf.reduce_max(w,axis=1,keepdims=True)
    
    check = w_s + w_max
    
    c = tf.constant(2)
    
    m = tf.multiply(tf.cast(c, tf.float64),w_max)
    
    d = tf.constant(0)
    
    equal1 =  tf.math.equal(check, m)
    equal1 = tf.cast(equal1,tf.float64)
    equal1_2 = Dot(axes=1)([equal1,w_s])    
    
    equal2 =  tf.math.equal(check, 0)
    equal2 = tf.cast(equal2,tf.float64)
    equal2_2 = Dot(axes=1)([equal1,w_s])
    

    
    e1 = tf.cast(equal1_2,tf.float64) + tf.cast(equal2_2,tf.float64) - w_max
    
    return e1

 

    
def get_accuracy(prediction, prob_test, test_sign):
    s = prediction.shape[0]
    print(s)
    mean_acc = 0
    inaccuracy = 0
    rmse = 0
    
    for t in range(s):
        
      
        prob = prob_test[t]
           
        #print(prob_test[t], test_sign[t], prediction[t])
        
        if ((prob - 0.5)*(prediction[t] - 0.5)) > 0:
            mean_acc = mean_acc + 1
        else:
            inaccuracy = inaccuracy + 1

        rmse = rmse + (prediction[t] - prob)**2

    rmse = math.sqrt(rmse/s)

    print("rmse DFlearn = ", rmse)
  
    return mean_acc, inaccuracy, rmse

#def Ranking(embedding_obj, num_items, l , m, dim, dim2, prob_data, idx):
def Ranking(embedding_obj, num_items, l , m, dim, dim2):

    #with open('my_dataset.pickle', 'rb') as data:
    #embeddings = pickle.load(data)
 
    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)
    train_data12 = np.copy(embedding_obj.train_data12)
    train_data22 = np.copy(embedding_obj.train_data22)
    validation_data1 = np.copy(embedding_obj.validation_data1)
    validation_data2 = np.copy(embedding_obj.validation_data2)

    #prob_data_orig = np.copy(prob_data)

    #score_orig = prob_data_orig[:,num_items]

    score_orig = np.copy(embedding_obj.score)

    
    prob_test = np.copy(embedding_obj.prob_test)
    prob_val = np.copy(embedding_obj.prob_val)
    prob_data = np.copy(embedding_obj.prob_data)
    rank_data1 = np.copy(embedding_obj.rank_data1)
    rank_data2 = np.copy(embedding_obj.rank_data2)
    rank_data12 = np.copy(embedding_obj.rank_data12)
    rank_data22 = np.copy(embedding_obj.rank_data22)

    items_ranked_original = np.copy(embedding_obj.items_ranked_original)

    sign_data = train_set1[:,-1]
    y_train = np.copy(train_set1[:,-1])
    y_train[y_train == -1] = 0
    y_test = np.copy(test_set1[:,-1])
    test_sign = test_set1[:,-1]
    y_test[y_test == -1] = 0

    
    train_val_sign = train_data12[:,-1]
    y_train_val = np.copy(train_data12[:,-1])
    y_train_val[y_train_val == -1] = 0
    y_val = np.copy(validation_data1[:,-1])
    val_sign = validation_data1[:,-1]
    y_val[y_val == -1] = 0

    total_pairs = int(comb(num_items,2))
    
    print("total_pairs = ", total_pairs)


    n_cycle = 0
    n_cycle2 = 0
    n_cycle3 = 0
    cycle_fraction = 0

        

    #hyperparameters
    #learning rate (0.0009 , 0.001)
    #kernel regularization parameter (0.0000095, 0.00009, 0.0009)
    #no. of embedding dimensions (10, 30, 50) (not used here)
    #no. of hidden nodes in the 1st layer (2*num_items, 8*num_items, 32*num_items)

    '''parameters = [(0.0009, 0.0009, 32*num_items), (0.0009, 0.00009, 32*num_items), (0.0009, 0.0000095, 32*num_items),(0.0009, 0.0009, 8*num_items), (0.0009, 0.00009, 8*num_items), (0.0009, 0.0000095, 8*num_items), (0.0009, 0.0009, 2*num_items), (0.0009, 0.00009, 2*num_items), (0.0009, 0.0000095, 2*num_items),
                  (0.001, 0.0009, 32*num_items), (0.001, 0.00009, 32*num_items), (0.001, 0.0000095, 32*num_items),(0.001, 0.0009, 8*num_items), (0.001, 0.00009, 8*num_items), (0.001, 0.0000095, 8*num_items), (0.001, 0.0009, 2*num_items), (0.001, 0.00009, 2*num_items), (0.001, 0.0000095, 2*num_items),]

    lr, c, hn = parameters[int(idx)]
    print("lr = ",lr)
    print("c = ",c)
    print("hn = ",hn)'''

    lr = 0.0009
    c = 0.00009
    hn = 32*num_items

    
    # Model.
    inputA = Input(shape=(num_items,), dtype="float64")
         
    h_1 = Dense(hn, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_1")(inputA)
    h_2 = Dense(32*dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_2")(h_1)
    #h_22 = Dense(16*dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_22")(h_2)
    h_u = Dense(dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_u")(h_2) #embeddings
    #h_u = tf.nn.l2_normalize(h_u, axis = 0) #normalized embeddings
    
    
    model1 = Model(inputs=inputA, outputs=h_u)


    input1 = Input(shape=(num_items,), dtype="float64")
    input2 = Input(shape=(num_items,), dtype="float64")

    # Create left and right twin models
    left_model = model1(input1)
    right_model = model1(input2)

    h_3 = Lambda(f)([left_model, right_model])

    s = Dense(1, activation="sigmoid")(h_3)

    # Build model.
    model = Model(inputs=[input1, input2], outputs=s)
    #model = Model(inputs=[inputA, inputB], outputs=s)
    
    keras.optimizers.Adam(learning_rate = lr, beta_1 = 0.9, beta_2=0.999, amsgrad=False)
    model.layers[4].set_weights([np.ones(model.layers[4].get_weights()[0].shape), np.zeros(model.layers[4].get_weights()[1].shape)])
    model.layers[4].trainable = False
    model.compile(optimizer="adam", loss="binary_crossentropy")
    keras.layers.Dropout(0.2, noise_shape=None, seed=None)
        
    
    # Train model.
    history1 = model.fit([train_set1[:,:-1],train_set2[:,:]], y_train, batch_size=8, epochs=100, validation_split = 0.3, verbose=1)

    #history2 = model.fit([train_data12[:,:-1],train_data22[:,:]], y_train_val, batch_size=8, epochs=50, validation_split = 0.3, verbose=1)               

    #evaluate on test data
    loss = model.evaluate([test_set1[:,:-1],test_set2[:,:]], y_test, batch_size=8)
    

    prediction = model.predict([test_set1[:,:-1],test_set2[:,:]])

    #prediction_val = model.predict([validation_data1[:,:-1],validation_data2[:,:]])
    
    prediction_embed2 = model.predict([rank_data12,rank_data22])
    prediction_embed3 = model.predict([rank_data22,rank_data12])

    #print("prediction_embed2 = ", prediction_embed2, "size = ", prediction_embed2.shape)
    
    prob_data3 = np.zeros((num_items,(num_items+1)))

    w = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if prediction_embed2[w][0] >= 0.5:
               prob_data3[i][j] = prediction_embed2[w][0]
               #prob_data3[j][i] = 1 - prediction_embed2[w][0] #or prob_data3[j][i] = prediction_embed3[w][0]
               
            else:
               prob_data3[j][i] = 1 - prediction_embed2[w][0]
               #prob_data3[i][j] = prediction_embed2[w][0] #or prob_data3[j][i] = 1 - prediction_embed3[w][0]
               

            w = w + 1


    w = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if prediction_embed2[w][0] > 0.5:
               prob_data3[i][num_items] = prob_data3[i][num_items] + 1
               
            elif prediction_embed2[w][0] < 0.5:
               prob_data3[j][num_items] = prob_data3[j][num_items] + 1

            w = w + 1

    
    accuracy, inaccuracy, rmse = get_accuracy(prediction, prob_test, test_sign)
    #accuracy_val, inaccuracy_val = get_accuracy(prediction_val, prob_val, val_sign)
    
    print("rmse = ", rmse)

    np.savetxt("Prob_DFlearn.txt", prediction)
    np.savetxt("prob_test.txt", prob_test)
    
    test_size = prediction.shape[0]
    accuracy = (accuracy/(test_size))
    inaccuracy = (inaccuracy/(test_size))

    '''val_size = prediction_val.shape[0]
    accuracy_val = (accuracy_val/(val_size))
    inaccuracy_val = (inaccuracy_val/(val_size))'''    
    
    
    print("l = ", l, "m = ", m, "dim2 = ", dim2)
    print("test_accuracy = ", accuracy, "test_inaccuracy = ", inaccuracy)
    #print("val_accuracy = ", accuracy_val, "val_inaccuracy = ", inaccuracy_val)
    
    score = np.zeros(num_items)
    keys = np.zeros(num_items)
    
    for i in range(num_items):
        score[i] = prob_data3[i][num_items]
        keys[i] = i

    

    d = dict(zip(keys, score))

    
    items_ranked_dflearn = []
    score_sorted_dflearn = []

    for k in sorted(d, key=d.get, reverse=True):
        items_ranked_dflearn.append(k)
        score_sorted_dflearn.append(d[k])

   
    A = 0
    D = 0

    original_rank = []
    dflearn_rank = []
    for i in range(num_items):
        original_rank.append(items_ranked_original[i])
        dflearn_rank.append(items_ranked_dflearn[i])

   
    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if (original_rank.index(i) - original_rank.index(j)) * (dflearn_rank.index(i) - dflearn_rank.index(j)) > 0 :
                A = A + 1
            else:
                D = D + 1
                
    ktc = (A - D)/total_pairs

    ktc2, p_val = stats.kendalltau(items_ranked_dflearn,items_ranked_original)
    print("ktc = ",ktc, "ktc2 = ", ktc2)

    ktc3, p_val = stats.kendalltau(score,score_orig)
    print("ktc3 = ",ktc3)


    #upset calculation
    
    upsets = 0
    
    test_data1 = test_set1[:,:-1]
    test_data2 = test_set2[:,:]

    s = test_data1.shape[0]
    #print("s = ", s)

    for w in range(s):
        #print("w = ", w)
        ind1 = np.argmax(test_data1[w,:])
        ind2 = np.argmax(test_data2[w,:])
        
        if (score[ind1] - score[ind2]) * (prob_test[w] - 0.5) < 0:
            upsets += 1

    upsets = upsets/s

    print("upsets = ", upsets)

    #3-cycles in the embedding
    cycle = []
    cycle2 = []
    cycle_frac = []
    n_cycle = 0
    n_cycle2 = 0
    cycle_fraction = 0
    
    
    
    for i in range(num_items):
        for j in range(num_items):
            if i == j:
                continue
            for k in range(num_items):
                if i == k or j == k:
                    continue
                if (prob_data[i][j] >= 0.5) and (prob_data[j][k] >= 0.5):
                    if (prob_data[k][i] >= 0.5):
                        
                #if ((prob_data[i][j] - 0.5) * (prob_data[j][k] - 0.5)) >= 0 and ((prob_data[i][j] - 0.5) * (prob_data[k][i] - 0.5)) >= 0:
                        n_cycle = n_cycle + 1
                        
                        #if ((prob_data3[i][j] - 0.5) * (prob_data3[j][k] - 0.5)) >= 0 and ((prob_data3[i][j] - 0.5) * (prob_data3[k][i] - 0.5)) >= 0:
                        
                        if (prob_data3[i][j] >= 0.5) and (prob_data3[j][k] >= 0.5):
                            if(prob_data3[k][i] >= 0.5):
                                n_cycle2 = n_cycle2 + 1


    cycle_fraction = n_cycle2/n_cycle
                                
    print("no. of cycles in original embedding = ", n_cycle)
    print("no. of cycles in learned embedding = ", n_cycle2)
    print("fraction of cycles captured = ", cycle_fraction)
    
    

       
    return accuracy, ktc3, upsets, n_cycle, cycle_fraction, rmse

def Ranking_SFData(embedding_obj, num_items, l , m, dim, dim2, c, embeddings):

    #with open('my_dataset.pickle', 'rb') as data:
    #embeddings = pickle.load(data)

    U = np.copy(embeddings)
    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)

    score_orig = np.copy(embedding_obj.score)

    
    prob_test = np.copy(embedding_obj.prob_test)
    prob_data = np.copy(embedding_obj.prob_data)
    rank_data1 = np.copy(embedding_obj.rank_data1)
    rank_data2 = np.copy(embedding_obj.rank_data2)
    rank_data12 = np.copy(embedding_obj.rank_data12)
    rank_data22 = np.copy(embedding_obj.rank_data22)

    items_ranked_original = np.copy(embedding_obj.items_ranked_original)

    sign_data = train_set1[:,-1]
    y_train = np.copy(train_set1[:,-1])
    y_train[y_train == -1] = 0
    y_test = np.copy(test_set1[:,-1])
    test_sign = test_set1[:,-1]
    y_test[y_test == -1] = 0

    total_pairs = int(comb(num_items,2))
    
    print("total_pairs = ", total_pairs)

    train_set12 = train_set1[:,:-1]
    train_set22 = train_set2[:,:]

    test_set12 = test_set1[:,:-1]
    test_set22 = test_set2[:,:]


    s = train_set12.shape[0]
    s2 = test_set12.shape[0]

    s_r1 = rank_data12.shape[0]
    
    U_rank1 = np.zeros((s_r1,(dim)))
    U_rank2 = np.zeros((s_r1,(dim)))

   
    U_train1 = np.zeros((int(m*l),(dim)))
    U_train2 = np.zeros((int(m*l),(dim)))

    U_test1 = np.zeros((int(int(comb(num_items,2))-m),(dim)))
    U_test2 = np.zeros((int(int(comb(num_items,2))-m),(dim)))
    
    for z in range(s):
        ind1 = np.argmax(train_set12[z,:])
        ind2 = np.argmax(train_set22[z,:])
        U_train1[z] = U[:,ind1]
        U_train2[z] = U[:,ind2]

    for z in range(s2):
        ind1 = np.argmax(test_set12[z,:])
        ind2 = np.argmax(test_set22[z,:])
        U_test1[z] = U[:,ind1]
        U_test2[z] = U[:,ind2]

    for z in range(s_r1):
        ind1 = np.argmax(rank_data12[z,:])
        ind2 = np.argmax(rank_data22[z,:])
        U_rank1[z] = U[:,ind1]
        U_rank2[z] = U[:,ind2]

    
    # Model.
    inputA = Input(shape=(num_items,), dtype="float64")
         
    h_1 = Dense(32*num_items, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_1")(inputA)
    h_2 = Dense(32*dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_2")(h_1)
    #h_22 = Dense(16*dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_22")(h_2)
    h_u = Dense(dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_u")(h_2) #embeddings
    #h_u_norm = tf.nn.l2_normalize(h_u, axis=-1) #normalized embeddings
    
    
    model1 = Model(inputs=inputA, outputs=h_u)


    input1 = Input(shape=(num_items,), dtype="float64")
    input2 = Input(shape=(num_items,), dtype="float64")

    # Create left and right twin models
    left_model = model1(input1)
    right_model = model1(input2)

    h_3 = Lambda(f)([left_model, right_model])

    s = Dense(1, activation="sigmoid")(h_3)

    # Build model.
    model = Model(inputs=[input1, input2], outputs=s)
    #model = Model(inputs=[inputA, inputB], outputs=s)
    
    keras.optimizers.Adam(learning_rate = 0.0009, beta_1 = 0.9, beta_2=0.999, amsgrad=False)
    model.layers[4].set_weights([np.ones(model.layers[4].get_weights()[0].shape), np.zeros(model.layers[4].get_weights()[1].shape)])
    model.layers[4].trainable = False
    model.compile(optimizer="adam", loss="binary_crossentropy")
    keras.layers.Dropout(0.2, noise_shape=None, seed=None)

    # Train model.
    history = model.fit([train_set1[:,:-1],train_set2[:,:]], y_train, batch_size=8, epochs=20, validation_split = 0.3, verbose=1)

    #evaluate on test data
    loss = model.evaluate([test_set1[:,:-1],test_set2[:,:]], y_test, batch_size=8)
    

    prediction = model.predict([test_set1[:,:-1],test_set2[:,:]])

    prediction_embed2 = model.predict([rank_data12,rank_data22])
    
    '''# Train model.
    history = model.fit([U_train1,U_train2], y_train, batch_size=8, epochs=100, validation_split = 0.3, verbose=1)

               

    #evaluate on test data
    loss = model.evaluate([U_test1,U_test2], y_test, batch_size=8)
    

    prediction = model.predict([U_test1,U_test2])
    
    prediction_embed2 = model.predict([U_rank1,U_rank2])'''
    #prediction_embed3 = model.predict([rank_data22,rank_data12])

    #print("prediction_embed2 = ", prediction_embed2, "size = ", prediction_embed2.shape)
    
    prob_data3 = np.zeros((num_items,(num_items+1)))

    w = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if prediction_embed2[w][0] >= 0.5:
               prob_data3[i][j] = prediction_embed2[w][0]
               #prob_data3[j][i] = 1 - prediction_embed2[w][0] #or prob_data3[j][i] = prediction_embed3[w][0]
               
            else:
               prob_data3[j][i] = 1 - prediction_embed2[w][0]
               #prob_data3[i][j] = prediction_embed2[w][0] #or prob_data3[j][i] = 1 - prediction_embed3[w][0]
               

            w = w + 1


    w = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if prediction_embed2[w][0] > 0.5:
               prob_data3[i][num_items] = prob_data3[i][num_items] + 1
               
            elif prediction_embed2[w][0] < 0.5:
               prob_data3[j][num_items] = prob_data3[j][num_items] + 1

            w = w + 1

    
    accuracy, inaccuracy, rmse = get_accuracy(prediction, prob_test, test_sign)
    #accuracy_val, inaccuracy_val = get_accuracy(prediction_val, prob_val, val_sign)
    
    print("rmse = ", rmse)
    
    
    test_size = prediction.shape[0]
    accuracy = (accuracy/(test_size))
    inaccuracy = (inaccuracy/(test_size))
    
    
    print("l = ", l, "m = ", m, "dim2 = ", dim2)
    print("test_accuracy = ", accuracy, "test_inaccuracy = ", inaccuracy)
    
    score = np.zeros(num_items)
    keys = np.zeros(num_items)
    
    for i in range(num_items):
        score[i] = prob_data3[i][num_items]
        keys[i] = i

    

    d = dict(zip(keys, score))

    
    items_ranked_dflearn = []
    score_sorted_dflearn = []

    for k in sorted(d, key=d.get, reverse=True):
        items_ranked_dflearn.append(k)
        score_sorted_dflearn.append(d[k])

   
    A = 0
    D = 0

    original_rank = []
    dflearn_rank = []
    for i in range(num_items):
        original_rank.append(items_ranked_original[i])
        dflearn_rank.append(items_ranked_dflearn[i])

   
    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if (original_rank.index(i) - original_rank.index(j)) * (dflearn_rank.index(i) - dflearn_rank.index(j)) > 0 :
                A = A + 1
            else:
                D = D + 1
                
    ktc = (A - D)/total_pairs

    ktc2, p_val = stats.kendalltau(items_ranked_dflearn,items_ranked_original)
    print("ktc = ",ktc, "ktc2 = ", ktc2)
    
    '''#3-cycles in the embedding
    cycle = []
    cycle2 = []
    cycle_frac = []
    n_cycle = 0
    n_cycle2 = 0
    cycle_fraction = 0
    
    
    

    for i in range(num_items):
        for j in range(num_items):
            if i == j:
                continue
            for k in range(num_items):
                if i == k or j == k:
                    continue
                if (prob_data[i][j] >= 0.5) and (prob_data[j][k] >= 0.5):
                    if (prob_data[k][i] >= 0.5):
                #if ((prob_data[i][j] - 0.5) * (prob_data[j][k] - 0.5)) >= 0 and ((prob_data[i][j] - 0.5) * (prob_data[k][i] - 0.5)) >= 0:
                        n_cycle = n_cycle + 1
                        cycle.append(n_cycle)
                        #print("prob_data[i][j] = ", prob_data[i][j], "prob_data[j][k] = ", prob_data[j][k], "prob_data[k][i] = ", prob_data[k][i])
                        #print("prob_data3[i][j] = ", prob_data3[i][j], "prob_data3[j][k] = ", prob_data3[j][k], "prob_data3[k][i] = ", prob_data3[k][i])
                        #if ((prob_data3[i][j] - 0.5) * (prob_data3[j][k] - 0.5)) >= 0 and ((prob_data3[i][j] - 0.5) * (prob_data3[k][i] - 0.5)) >= 0:
                        if (prob_data3[i][j] >= 0.5) and (prob_data3[j][k] >= 0.5):
                            if(prob_data3[k][i] >= 0.5):
                                n_cycle2 = n_cycle2 + 1
                                cycle_fraction = n_cycle2/n_cycle
                                cycle2.append(n_cycle2)
                                cycle_frac.append(cycle_fraction)
                                #print("prob_data[i][j] = ", prob_data[i][j], "prob_data[j][k] = ", prob_data[j][k], "prob_data[k][i] = ", prob_data[k][i])
                                #print("prob_data3[i][j] = ", prob_data3[i][j], "prob_data3[j][k] = ", prob_data3[j][k], "prob_data3[k][i] = ", prob_data3[k][i])

    print("no. of cycles in original embedding = ", n_cycle)
    print("no. of cycles in learned embedding = ", n_cycle2)
    print("fraction of cycles captured = ", cycle_fraction)'''

    ktc3, p_val = stats.kendalltau(score,score_orig)
    print("ktc3 = ",ktc3)    

       
    return accuracy, ktc3, rmse



def Ranking_BTLdata(embedding_obj, num_items, l , m, dim, dim2, c):

    #with open('my_dataset.pickle', 'rb') as data:
    #embeddings = pickle.load(data)
 
    train_set1 = np.copy(embedding_obj.train_data1)
    train_set2 = np.copy(embedding_obj.train_data2)
    test_set1 = np.copy(embedding_obj.test_data1)
    test_set2 = np.copy(embedding_obj.test_data2)

    
    prob_test = np.copy(embedding_obj.prob_test)
    
    
    rank_data12 = np.copy(embedding_obj.rank_data12)
    rank_data22 = np.copy(embedding_obj.rank_data22)

    items_ranked_original = np.copy(embedding_obj.items_ranked_original)
    score_orig = np.copy(embedding_obj.score)

    sign_data = train_set1[:,-1]
    y_train = np.copy(train_set1[:,-1])
    y_train[y_train == -1] = 0
    y_test = np.copy(test_set1[:,-1])
    test_sign = test_set1[:,-1]
    y_test[y_test == -1] = 0

    total_pairs = int(comb(num_items,2))
    
    print("total_pairs = ", total_pairs)  

    
    # Model.
    inputA = Input(shape=(num_items,), dtype="float64")
         
    h_1 = Dense(32*num_items, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_1")(inputA)
    h_2 = Dense(32*dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_2")(h_1)
    #h_22 = Dense(16*dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_22")(h_2)
    h_u = Dense(dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_u")(h_2) #embeddings
    #h_u_norm = tf.nn.l2_normalize(h_u, axis=-1) #normalized embeddings
    
    
    model1 = Model(inputs=inputA, outputs=h_u)


    input1 = Input(shape=(num_items,), dtype="float64")
    input2 = Input(shape=(num_items,), dtype="float64")

    # Create left and right twin models
    left_model = model1(input1)
    right_model = model1(input2)

    h_3 = Lambda(f)([left_model, right_model])

    s = Dense(1, activation="sigmoid")(h_3)

    # Build model.
    model = Model(inputs=[input1, input2], outputs=s)
    #model = Model(inputs=[inputA, inputB], outputs=s)
    
    keras.optimizers.Adam(learning_rate = 0.0009, beta_1 = 0.9, beta_2=0.999, amsgrad=False)
    model.layers[4].set_weights([np.ones(model.layers[4].get_weights()[0].shape), np.zeros(model.layers[4].get_weights()[1].shape)])
    model.layers[4].trainable = False
    model.compile(optimizer="adam", loss="binary_crossentropy")
    keras.layers.Dropout(0.2, noise_shape=None, seed=None)
        
    
    # Train model.
    history = model.fit([train_set1[:,:-1],train_set2[:,:]], y_train, batch_size=8, epochs=10, validation_split = 0.3, verbose=1)

               

    #evaluate on test data
    loss = model.evaluate([test_set1[:,:-1],test_set2[:,:]], y_test, batch_size=8)
    

    prediction = model.predict([test_set1[:,:-1],test_set2[:,:]])
    
    prediction_embed2 = model.predict([rank_data12,rank_data22])
    #prediction_embed3 = model.predict([rank_data22,rank_data12])

    #print("prediction_embed2 = ", prediction_embed2, "size = ", prediction_embed2.shape)
    
    prob_data3 = np.zeros((num_items,(num_items+1)))


    w = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if prediction_embed2[w][0] >= 0.5:
               prob_data3[i][j] = prediction_embed2[w][0]
               #prob_data3[j][i] = 1 - prediction_embed2[w][0] #or prob_data3[j][i] = prediction_embed3[w][0]
               
            else:
               prob_data3[j][i] = 1 - prediction_embed2[w][0]
               #prob_data3[i][j] = prediction_embed2[w][0] #or prob_data3[j][i] = 1 - prediction_embed3[w][0]
               

            w = w + 1


    w = 0

    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if prediction_embed2[w][0] > 0.5:
               prob_data3[i][num_items] = prob_data3[i][num_items] + 1
               
            elif prediction_embed2[w][0] < 0.5:
               prob_data3[j][num_items] = prob_data3[j][num_items] + 1

            w = w + 1

    
    accuracy, inaccuracy, rmse = get_accuracy(prediction, prob_test, test_sign)
    
    
    test_size = prediction.shape[0]
    accuracy = (accuracy/(test_size))
    inaccuracy = (inaccuracy/(test_size))
    
    
    print("l = ", l, "m = ", m, "dim2 = ", dim2)
    print("test_accuracy = ", accuracy, "test_inaccuracy = ", inaccuracy, "rmse = ", rmse)
    
    score = np.zeros(num_items)
    keys = np.zeros(num_items)
    
    for i in range(num_items):
        score[i] = prob_data3[i][num_items]
        keys[i] = i

    

    d = dict(zip(keys, score))

    
    items_ranked_dflearn = []
    score_sorted_dflearn = []

    for k in sorted(d, key=d.get, reverse=True):
        items_ranked_dflearn.append(k)
        score_sorted_dflearn.append(d[k])

   
    A = 0
    D = 0

    original_rank = []
    dflearn_rank = []
    for i in range(num_items):
        original_rank.append(items_ranked_original[i])
        dflearn_rank.append(items_ranked_dflearn[i])

   
    for i in range(num_items-1):
        for j in range(i+1, num_items):
            if (original_rank.index(i) - original_rank.index(j)) * (dflearn_rank.index(i) - dflearn_rank.index(j)) > 0 :
                A = A + 1
            else:
                D = D + 1
                
    ktc = (A - D)/total_pairs

    ktc2, p_val = stats.kendalltau(items_ranked_dflearn,items_ranked_original)
    print("ktc = ",ktc, "ktc2 = ", ktc2)

    ktc3, p_val = stats.kendalltau(score,score_orig)
    print("ktc3 = ",ktc3)    

       
    return accuracy, ktc3, rmse

