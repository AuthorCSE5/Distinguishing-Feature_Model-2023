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

 

    
def get_accuracy(prediction, test1, test2, prob_data):
    s = prediction.shape[0]
    print(s)
    mean_acc = 0
    inaccuracy = 0
    rmse = 0
    
    for t in range(s):
        ind1 = np.argmax(test1[t,:])
        ind2 = np.argmax(test2[t,:])
    
        if ((prob_data[ind1][ind2] - 0.5)*(prediction[t] - 0.5)) > 0:
            mean_acc = mean_acc + 1
        else:
            inaccuracy = inaccuracy + 1

        rmse = rmse + (prediction[t] - prob_data[ind1][ind2])**2

        

    rmse = math.sqrt(rmse/s)
  
    print("rmse DFlearn = ", rmse)

    
  
    return mean_acc, inaccuracy, rmse


def Ranking(train_data1, train_data2, test_data1, test_data2, num_items, dim2, c, prob_data):

    #with open('my_dataset.pickle', 'rb') as data:
    #embeddings = pickle.load(data)
 
    train_set1 = np.copy(train_data1)
    train_set2 = np.copy(train_data2)
    test_set1 = np.copy(test_data1)
    test_set2 = np.copy(test_data2)
        
    prob_data = np.copy(prob_data)
    

    sign_data = train_set1[:,-1]
    y_train = np.copy(train_set1[:,-1])
    y_train[y_train == -1] = 0
    y_test = np.copy(test_set1[:,-1])
    test_sign = test_set1[:,-1]
    y_test[y_test == -1] = 0

    # Model.
    inputA = Input(shape=(num_items,), dtype="float64")
         
    '''h_1 = Dense(num_items, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_1")(inputA)
    d_1 = Dropout(0.2)(h_1)
    h_2 = Dense(32*dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_2")(d_1)
    d_2 = Dropout(0.5)(h_2)
    #h_22 = Dense(16*dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_22")(h_2)
    h_u = Dense(dim2, activation="relu", kernel_regularizer=regularizers.l2(c), name = "h_u")(d_2) #embeddings
    #h_u_norm = tf.nn.l2_normalize(h_u, axis=-1) #normalized embeddings'''

    h_1 = Dense(num_items, activation="relu", name = "h_1")(inputA)
    d_1 = Dropout(0.2)(h_1)
    h_2 = Dense(32*dim2, activation="relu", name = "h_2")(d_1)
    d_2 = Dropout(0.7)(h_2)
    #h_22 = Dense(16*dim2, activation="relu", name = "h_22")(h_2)
    h_u = Dense(dim2, activation="relu", name = "h_u")(d_2) #embeddings
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
    
    keras.optimizers.SGD(learning_rate = 0.09, momentum = 0.99)#, amsgrad=False)
    model.layers[4].set_weights([np.ones(model.layers[4].get_weights()[0].shape), np.zeros(model.layers[4].get_weights()[1].shape)])
    model.layers[4].trainable = False
    model.compile(optimizer="sgd", loss="binary_crossentropy")
    #keras.layers.Dropout(0.2, noise_shape=None, seed=None)
        
    
    # Train model.
    history = model.fit([train_set1[:,:-1],train_set2[:,:]], y_train, batch_size=32, epochs=200, validation_split = 0.3, verbose=1)

               

    #evaluate on test data
    loss = model.evaluate([test_set1[:,:-1],test_set2[:,:]], y_test, batch_size=8)
    

    prediction = model.predict([test_set1[:,:-1],test_set2[:,:]])

    test1 = test_set1[:,:-1]
    test2 = test_set2[:,:]
       
    accuracy, inaccuracy, rmse = get_accuracy(prediction, test1, test2, prob_data)
    
    
    test_size = prediction.shape[0]
    accuracy = (accuracy/(test_size))
    inaccuracy = (inaccuracy/(test_size))
    
    print("test_accuracy = ", accuracy, "test_inaccuracy = ", inaccuracy)
    
    
    return accuracy, rmse
