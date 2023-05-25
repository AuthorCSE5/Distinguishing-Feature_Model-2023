import numpy as np
from numpy import random
#from math import comb, log


def generate_embedding(num_items, dim, std_deviation, t):

        np.random.seed(t)
        
        U = np.random.normal(0, scale = std_deviation, size = (dim,num_items))

        U = U/np.linalg.norm(U,2,axis = 0)

        #U = U/np.sum(U)

        return U

def generate_embedding_new(num_items, dim, std_deviation, t):

        np.random.seed(t)

        num_items_block1 = num_items * (2/3)
        num_items_block2 = num_items - int(num_items_block1)
        num_items_block2_new = int(num_items_block2 * (2/3))
        num_items_block3 = num_items_block2 - num_items_block2_new
        num_items_block2 = num_items_block2_new

        x1 = t
        size1 = int(num_items_block1/3) #+ 1
        size2 = int(num_items_block1/3) #+ 1
        
        U1 = np.random.normal(x1, scale = std_deviation, size = (1, size1))
        print("U1 = ", U1)

        x2 = t
        U2 = np.random.normal(x2, scale = std_deviation, size = (1, size1))
        print("U2 = ", U2)
        U = np.append(U1, U2, axis = 0)
        print("U = ", U)

        x3 = t        
        U3 = np.random.normal(x3, scale = std_deviation, size = (1, size1))
        U = np.append(U, U3, axis = 0)
        print("U = ", U.shape)

        
        x1 = x1 - 8      
        V1 = np.random.normal(x1, scale = std_deviation, size = (1, size2))

        x2 = x2 + 4        
        V2 = np.random.normal(x2, scale = std_deviation, size = (1, size2))
        V = np.append(V1, V2, axis = 0)
        print("V = ", V)

        x3 = x3 + 6        
        V3 = np.random.normal(x3, scale = std_deviation, size = (1, size2))
        V = np.append(V, V3, axis = 0)
        print("V = ", V.shape)

        x1 = x1 + 3      
        W1 = np.random.normal(x1, scale = std_deviation, size = (1, int(num_items_block1) - (size1 + size2)))

        x2 = x2 - 9        
        W2 = np.random.normal(x2, scale = std_deviation, size = (1, int(num_items_block1) - (size1 + size2)))
        W = np.append(W1, W2, axis = 0)
        print("W = ", W)

        x3 = x3 + 4        
        W3 = np.random.normal(x3, scale = std_deviation, size = (1, int(num_items_block1) - (size1 + size2)))
        W = np.append(W, W3, axis = 0)
        print("W = ", W.shape)

        T = np.append(U, V, axis = 1)
        print("T = ", T.shape)
        T = np.append(T, W, axis = 1)
        print("T = ", T.shape)

        print("num_items_block1 = ", int(num_items_block1))

        x1 = t - 10
        size1 = int(num_items_block2/3)
        size2 = int(num_items_block2/3)
        
        U1 = np.random.normal(x1, scale = std_deviation, size = (1, size1))
        print("U1 = ", U1)

        x2 = t - 10
        U2 = np.random.normal(x2, scale = std_deviation, size = (1, size1))
        print("U2 = ", U2)
        U = np.append(U1, U2, axis = 0)
        print("U = ", U)

        x3 = t - 10        
        U3 = np.random.normal(x3, scale = std_deviation, size = (1, size1))
        U = np.append(U, U3, axis = 0)
        print("U = ", U.shape)

        
        x1 = x1 - 8      
        V1 = np.random.normal(x1, scale = std_deviation, size = (1, size2))

        x2 = x2 + 4        
        V2 = np.random.normal(x2, scale = std_deviation, size = (1, size2))
        V = np.append(V1, V2, axis = 0)
        print("V = ", V)

        x3 = x3 + 6        
        V3 = np.random.normal(x3, scale = std_deviation, size = (1, size2))
        V = np.append(V, V3, axis = 0)
        print("V = ", V.shape)

        x1 = x1 + 3      
        W1 = np.random.normal(x1, scale = std_deviation, size = (1, int(num_items_block2) - (size1 + size2)))

        x2 = x2 - 9        
        W2 = np.random.normal(x2, scale = std_deviation, size = (1, int(num_items_block2) - (size1 + size2)))
        W = np.append(W1, W2, axis = 0)
        print("W = ", W)

        x3 = x3 + 4        
        W3 = np.random.normal(x3, scale = std_deviation, size = (1, int(num_items_block2) - (size1 + size2)))
        W = np.append(W, W3, axis = 0)
        print("W = ", W.shape)

        T2 = np.append(U, V, axis = 1)
        print("T2 = ", T2.shape)
        T2 = np.append(T2, W, axis = 1)
        print("T2 = ", T2.shape)

        T = np.append(T, T2, axis = 1)
        print("T = ", T.shape)

        

        x1 = t - 20
        size1 = int(num_items_block3/3)
        size2 = int(num_items_block3/3)
        
        U1 = np.random.normal(x1, scale = std_deviation, size = (1, size1))
        print("U1 = ", U1)

        x2 = t - 20
        U2 = np.random.normal(x2, scale = std_deviation, size = (1, size1))
        print("U2 = ", U2)
        U = np.append(U1, U2, axis = 0)
        print("U = ", U)

        x3 = t - 20        
        U3 = np.random.normal(x3, scale = std_deviation, size = (1, size1))
        U = np.append(U, U3, axis = 0)
        print("U = ", U.shape)

        
        x1 = x1 - 8      
        V1 = np.random.normal(x1, scale = std_deviation, size = (1, size2))

        x2 = x2 + 4        
        V2 = np.random.normal(x2, scale = std_deviation, size = (1, size2))
        V = np.append(V1, V2, axis = 0)
        print("V = ", V)

        x3 = x3 + 6        
        V3 = np.random.normal(x3, scale = std_deviation, size = (1, size2))
        V = np.append(V, V3, axis = 0)
        print("V = ", V.shape)

        x1 = x1 + 3      
        W1 = np.random.normal(x1, scale = std_deviation, size = (1, int(num_items_block3) - (size1 + size2)))

        x2 = x2 - 9        
        W2 = np.random.normal(x2, scale = std_deviation, size = (1, int(num_items_block3) - (size1 + size2)))
        W = np.append(W1, W2, axis = 0)
        print("W = ", W)

        x3 = x3 + 4        
        W3 = np.random.normal(x3, scale = std_deviation, size = (1, int(num_items_block3) - (size1 + size2)))
        W = np.append(W, W3, axis = 0)
        print("W = ", W.shape)

        T2 = np.append(U, V, axis = 1)
        print("T2 = ", T2.shape)
        T2 = np.append(T2, W, axis = 1)
        print("T2 = ", T2.shape)

        T = np.append(T, T2, axis = 1)
        print("T = ", T.shape)

        

        #E = np.ones((dim - 3,num_items))

        #T = np.append(T, E, axis = 0)
        #print("T = ", T.shape)

        #U = U/np.linalg.norm(U,2,axis = 0)

        #U = U/np.sum(U)

        print("t final = ", t)

        return T

def generate_embedding_uniform(num_items, dim, t):

        np.random.seed(t)
        
        U = np.random.uniform(low = -dim, high = dim, size = (dim,num_items))

        return U

def generate_clustered_embedding(num_items, dim, num_clusters):
        U = np.zeros((dim,num_items))
        
        cov = (1/np.sqrt(dim))*np.eye(dim,dim)

        clust_to_center = [1/ np.sqrt(np.sqrt(dim))*np.random.randn(dim, 1).reshape(dim,1).T[0] for i in range(num_clusters)]

        for i in range(num_items):
            clust_mean = clust_to_center[np.random.randint(0,num_clusters)]
            U[:, i] = np.random.multivariate_normal(clust_mean, cov).T

        return U
