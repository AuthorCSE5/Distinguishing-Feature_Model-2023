import numpy as np
import math
from scipy.special import comb

class embedding(object):
    #def __init__(self, num_items, dim, std_deviation, l, m):
    def __init__(self, embeddings, w, num_items, l, m):

        '''self.num_items = num_items
        self.dim = dim
        self.std_deviation = std_deviation
        self.l = l'''
        
        '''U = self._generate_embedding(num_items, dim, std_deviation)
        self.U = U'''
        
        prob, r, train_data1, train_data2, test_data1, test_data2, prob_test, pairs_train0, pairs_train1 = self._generate_data(embeddings, w, num_items, l, m)
        self.prob = prob
        self.r = r
        self.train_data1 = train_data1
        self.test_data1 = test_data1
        self.train_data2 = train_data2
        self.test_data2 = test_data2
        self.prob_test = prob_test
        self.pairs_train0 = pairs_train0
        self.pairs_train1 = pairs_train1

        prob_data, items_ranked_original, rank_data1, rank_data2, rank_data12, rank_data22, score = self._generate_ranking_data(embeddings, w, num_items)
        self.prob_data = prob_data
        #self.prob_data_test = prob_data_test
        self.rank_data1 = rank_data1
        self.rank_data2 = rank_data2
        self.rank_data12 = rank_data12
        self.rank_data22 = rank_data22
        self.items_ranked_original = items_ranked_original
        self.score = score
        
        

    '''def _generate_embedding(self,num_items, dim, std_deviation):
        
        U = np.random.normal(0, scale = std_deviation, size = (dim,num_items))

        return U'''

    def _generate_data(self, embeddings, w, num_items, l, m):

        U = np.copy(embeddings)
        w_proj = np.copy(w)
    
        train_data1 = np.zeros((int(m*l),(num_items+1)))
        train_data2 = np.zeros((int(m*l),(num_items)))
        
        test_data1 = np.zeros((int(int(comb(num_items,2))-m),(num_items+1)))
        test_data2 = np.zeros((int(int(comb(num_items,2))-m),(num_items)))

        pairs_1 = np.zeros((2 , int(comb(num_items,2))))
        pairs_train = np.zeros((2 , m))
        pairs_test = np.zeros((2 , int(comb(num_items,2)) - m))
        
        k = 0
        for i in range(num_items - 1):
            for j in range(i+1, num_items):
                pairs_1[0][k] = i
                pairs_1[1][k] = j
                k = k + 1

        np.random.shuffle(np.transpose(pairs_1))
        
        for i in range(m):
            pairs_train[0][i] = pairs_1[0][i]
            pairs_train[1][i] = pairs_1[1][i]

        for i in range(m,int(comb(num_items,2))):
            pairs_test[0][i-m] = pairs_1[0][i]
            pairs_test[1][i-m] = pairs_1[1][i]
            
        pairs_train0 = pairs_train[0][:].astype(int)
        pairs_train1 = pairs_train[1][:].astype(int)
        pairs_test0 = pairs_test[0][:].astype(int)
        pairs_test1 = pairs_test[1][:].astype(int)

        prob_test2 = np.zeros((m,1))

        w = 0
        #for i in range(num_items - 1):
         #   for j in range(i+1, num_items):
        for i in range(m):

            diff = np.abs(U[:,pairs_train0[i]] - U[:,pairs_train1[i]])
            diff2 = np.amax(diff)
            ind = np.argmax(diff)
            indices = np.argwhere(diff == np.amax(diff))
            #print("indices = ", indices)
            ind2 = indices.flatten().tolist()
            #print("ind2 = ", ind2)
            #prob = 1/(1+np.exp(-diff2))

            diff3 = (U[ind,pairs_train0[i]] - U[ind,pairs_train1[i]]) * w_proj[ind,0]
            prob = 1/(1+np.exp(-diff3))
            prob_test2[i][0] = prob

            r = np.random.rand(l)

            for k in range(l):
                
                #train_data[w][pairs_train1[i]] = 1
                
                if r[k] <= prob:                                        
                   train_data1[w][num_items] = 1
                   train_data1[w][pairs_train0[i]] = 1
                   train_data2[w][pairs_train1[i]] = 1
                elif r[k] > prob and r[k] <= 1:
                   train_data1[w][num_items] = -1
                   train_data1[w][pairs_train0[i]] = 1
                   train_data2[w][pairs_train1[i]] = 1
                   '''train_data[w][num_items] = -1
                   train_data[w][pairs_train0[i]] = 1
                   train_data[w][pairs_train1[i]] = -1'''
                   
                '''if (U[ind,pairs_train0[i]] >= U[ind,pairs_train1[i]]):
                   train_data[w][pairs_train0[i]] = 1
                   train_data[w][pairs_train1[i]] = -1
                else:
                   train_data[w][pairs_train0[i]] = -1
                   train_data[w][pairs_train1[i]] = 1'''
               
                #print(w, k, pairs_train0, pairs_train1)
                w = w + 1
                    

        w = 0

        prob_test = np.zeros((int(comb(num_items,2))-m,1))

        for i in range(int(comb(num_items,2)) - m):
            
                diff = np.abs(U[:,pairs_test0[i]] - U[:,pairs_test1[i]])
                #diff2 = np.amax(diff)
                ind = np.argmax(diff)
                indices = np.argwhere(diff == np.amax(diff))
                #print("indices_test = ", indices)
                ind2 = indices.flatten().tolist()
                #print("ind2_test = ", ind2)
                diff3 = (U[ind,pairs_test0[i]] - U[ind,pairs_test1[i]]) * w_proj[ind,0]
                #prob_test[i][0] = 1/(1+np.exp(-diff2))
                prob_test[i][0] = 1/(1+np.exp(-diff3))

                '''if (diff3 >= 0):
                   test_data[w][pairs_test0[i]] = 1
                   test_data[w][pairs_test1[i]] = -1
                   test_data[w][num_items] = 1
                else:
                   test_data[w][pairs_test0[i]] = 1
                   test_data[w][pairs_test1[i]] = -1
                   test_data[w][num_items] = -1'''

                if (diff3 >= 0):
                   '''test_data1[w][pairs_test0[i]] = 1
                   test_data2[w][pairs_test1[i]] = 1
                   test_data1[w][num_items] = 1'''
                   
                   test_data1[w][pairs_test0[i]] = 1
                   test_data2[w][pairs_test1[i]] = 1
                   test_data1[w][num_items] = 1
                else:
                   test_data1[w][pairs_test0[i]] = 1
                   test_data2[w][pairs_test1[i]] = 1
                   test_data1[w][num_items] = -1
                   
                 
                   
                s1 = test_data1.shape[0]
                #print("s1 = ", s1)

                s2 = prob_test.shape[0]
                #print("s2 = ", s2)

                
                   

                w = w + 1
                
        return prob, r, train_data1, train_data2, test_data1, test_data2, prob_test, pairs_train0, pairs_train1

    
    def _generate_ranking_data(self, embeddings, w, num_items):

        U = np.copy(embeddings)
        w_proj = np.copy(w)

        total_pair = int(comb(num_items,2))

        
        score = np.zeros(num_items)
        keys = np.zeros(num_items)

        prob_data = np.zeros((num_items,(num_items+1)))
        prob_data_test = np.zeros((num_items,num_items))
        rank_data1 = np.zeros((num_items-1,num_items))
        rank_data2 = np.zeros((num_items-1,num_items))
        rank_data12 = np.zeros((int(comb(num_items,2)),num_items))
        rank_data22 = np.zeros((int(comb(num_items,2)),num_items))

        w2 = 0
               
        i = 0
        for j in range(i+1,num_items):
            rank_data1[w2][i] = 1
            rank_data2[w2][j] = 1
            
            w2 = w2 + 1

        w2 = 0
                
        for i in range(num_items-1):
            for j in range(i+1, num_items):
                rank_data12[w2][i] = 1
                rank_data22[w2][j] = 1
            
                w2 = w2 + 1

        #prob_embedding = np.zeros((num_items, num_items))

        #print(U)
        
        for i in range(num_items-1):
            for j in range(i+1, num_items):
        
                diff = np.abs(U[:,i] - U[:,j])
                diff2 = np.amax(diff)
                ind = np.argmax(diff)
                diff3 = (U[ind,i] - U[ind,j]) * w_proj[ind,0]
                prob = 1/(1+np.exp(-diff3))
                
                
                #print(prob)

                
                
                
                
                
                if prob >= 0.5:
                   prob_data[i][j] = prob
                   #prob_data[i][num_items] = prob_data[i][num_items] + 1
                   
                else:
                   prob_data[j][i] = 1 - prob
                   #prob_data[j][num_items] = prob_data[j][num_items] + 1



                if prob > 0.5:
                   
                   prob_data[i][num_items] = prob_data[i][num_items] + 1
                   
                elif prob < 0.5:
                   
                   prob_data[j][num_items] = prob_data[j][num_items] + 1

                
        #print(prob_data)
        prob_data_test = prob_data[:,:-1]

        for i in self.pairs_train0:
            for j in self.pairs_train1:
                prob_data_test[i][j] = 0
                prob_data_test[j][i] = 0
        

        

        for i in range(num_items):
            score[i] = prob_data[i][num_items] #np.dot(U[:,i], w_proj)
            keys[i] = i

           
       

        d = dict(zip(keys, score))
        

        items_ranked_original = []
        score_sorted_original = []

        for k in sorted(d, key=d.get, reverse=True):
            items_ranked_original.append(k)
            score_sorted_original.append(d[k])

        
              

        return prob_data, items_ranked_original, rank_data1, rank_data2, rank_data12, rank_data22, score          

                        

        
                        


#d = embedding(4, 3, 1, 3, 4)
#print(d.rank_data1)
#print(d.rank_data2)
#print(d.U)
#print(d.prob)
#print(d.r)
#print(d.prob_test)
#print(d.train_data.shape[0])
#print(d.test_data.shape[0])
#for t in range(d.test_data.shape[0]):
#    print(t)
#    print(np.abs(d.test_data[:,:-1][t,-1]))'''

        
