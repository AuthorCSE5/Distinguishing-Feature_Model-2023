import sys
sys.path.append('../../src/')

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
from stellargraph import StellarDiGraph

from extract_network import extract_network

df = np.loadtxt("664_0_DFData.txt", dtype = float)  ##df = np.loadtxt("WoL_modifiedtrain0.txt", dtype = float)
df = df.astype(float)
print(df)

df_2 = df.copy()
df1 = pd.DataFrame(df_2, columns = ['a', 'b', 'wins'])
print(df1)

#df1['wins'] = df1['na'] - df1['nb']
print("min = ", min(df1['wins']))
#df1['wins'] = df1['wins'] - min(df1['wins'])
#df1 = df1.drop(df1[df1['wins'] <= 0].index)s


df1.columns = ['source', 'target', 'weight']

'''nodes1 = df1[['source']].copy()

print(type(nodes1))
nodes1 = nodes1.to_numpy()
print(nodes1)

nodes2 = df1[['target']].copy()

print(type(nodes2))
nodes2 = nodes2.to_numpy()
print(nodes2)

nodes = np.append(nodes1,nodes2,axis = 0)
#nodes = np.unique(nodes, axis = 0)

print("nodes = ", nodes)'''


'''data1 = df1[['a', 'b', 'na']].copy()
data1.columns=['source', 'target', 'weight']
data2 = df1[['b', 'a', 'nb']].copy()
data2.columns=['source', 'target', 'weight']
data = pd.concat([data1,data2], ignore_index=True)
print(data)'''

print(df1)

#df1['weight'] += 0.1
#print("data_weight_151429 = ", data.iloc[151429,2])
#print("data_weight_151439 = ", data.iloc[151639,2])
graph = StellarDiGraph(edges=df1)

print(graph.info())

adj = graph.to_adjacency_matrix(weighted=True)
assert np.abs(adj-adj.transpose()).sum() != 0
print(adj)
print(type(adj))
#np.savetxt("adjacency.txt", np.array(adj))

node_list = list(graph.nodes())

print(len(node_list))
'''
adj, node_list = extract_network(adj, np.array(node_list), lowest_degree=1)
print(len(node_list))'''

folder_name = './'

np.save(folder_name+'node_list', node_list)
sp.save_npz(folder_name+'adj', adj)
