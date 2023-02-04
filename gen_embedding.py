import numpy as np
from numpy import random
#from math import comb, log


def generate_embedding(num_items, dim, std_deviation, t):

        np.random.seed(t)
        
        U = np.random.normal(0, scale = std_deviation, size = (dim,num_items))

        U = U/np.linalg.norm(U,2,axis = 0)

        #U = U/np.sum(U)

        return U

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
