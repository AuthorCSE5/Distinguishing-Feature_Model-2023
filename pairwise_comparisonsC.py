import numpy as np
import math
from scipy.special import comb

class embedding(object):
    #def __init__(self, num_items, dim, std_deviation, l, m):
    def __init__(self, U, num_items, l, m, t):

        '''self.num_items = num_items
        self.dim = dim
        self.std_deviation = std_deviation
        self.l = l'''
        
        '''U = self._generate_embedding(num_items, dim, std_deviation)
        self.U = U'''
        
        prob, r, train_data1, train_data2, train_data12, train_data22, validation_data1, validation_data2, test_data1, test_data2, prob_test, prob_val, pairs_train0, pairs_train1 = self._generate_data(U, num_items, l, m)
        self.prob = prob
        self.r = r
        self.train_data1 = train_data1
        self.test_data1 = test_data1
        self.train_data2 = train_data2
        self.test_data2 = test_data2
        self.prob_test = prob_test
        self.prob_val = prob_val
        self.pairs_train0 = pairs_train0
        self.pairs_train1 = pairs_train1
        self.train_data12 = train_data12
        self.train_data22 = train_data22
        self.validation_data1 = validation_data1
        self.validation_data2 = validation_data2

        prob_data_test, items_ranked_original, rank_data1, rank_data2, rank_data12, rank_data22, prob_data, score = self._generate_ranking_data(U, num_items)
        #self.prob_data = prob_data
        self.prob_data_test = prob_data_test
        self.rank_data1 = rank_data1
        self.rank_data2 = rank_data2
        self.rank_data12 = rank_data12
        self.rank_data22 = rank_data22
        self.items_ranked_original = items_ranked_original
        self.score = score
        self.prob_data = prob_data
        
        

    '''def _generate_embedding(self,num_items, dim, std_deviation):
        
        U = np.random.normal(0, scale = std_deviation, size = (dim,num_items))

        return U'''

    def _generate_data(self, U, num_items, l, m):

        features = np.copy(U)
    
        train_data1 = np.zeros((int(m*l),(num_items+1)))
        train_data2 = np.zeros((int(m*l),(num_items)))
        
        test_data1 = np.zeros((int(int(comb(num_items,2))-m),(num_items+1)))
        test_data2 = np.zeros((int(int(comb(num_items,2))-m),(num_items)))

        pairs_1 = np.zeros((2 , int(comb(num_items,2))))
        pairs_train = np.zeros((2 , m))
        pairs_test = np.zeros((2 , int(comb(num_items,2)) - m))

        data_size = np.shape(train_data1)[0]
        train_size = int(data_size * 0.75)
        validation_size = data_size - train_size

        prob_train = np.zeros((data_size,1))
        
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


        w = 0
        #for i in range(num_items - 1):
         #   for j in range(i+1, num_items):
        for i in range(m):

            diff = np.abs(features[:,pairs_train0[i]] - features[:,pairs_train1[i]])
            diff2 = np.amax(diff)
            ind = np.argmax(diff)
            indices = np.argwhere(diff == np.amax(diff))
            #print("indices = ", indices)
            ind2 = indices.flatten().tolist()
            #print("ind2 = ", ind2)
            #prob = 1/(1+np.exp(-diff2))

            diff3 = features[ind,pairs_train0[i]] - features[ind,pairs_train1[i]]
            prob = 1/(1+np.exp(-diff3))

            r = np.random.rand(l)

            for k in range(l):

                prob_train[w][0] = prob
                
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
            
                diff = np.abs(features[:,pairs_test0[i]] - features[:,pairs_test1[i]])
                #diff2 = np.amax(diff)
                ind = np.argmax(diff)
                indices = np.argwhere(diff == np.amax(diff))
                #print("indices_test = ", indices)
                ind2 = indices.flatten().tolist()
                #print("ind2_test = ", ind2)
                diff3 = (features[ind,pairs_test0[i]] - features[ind,pairs_test1[i]])
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
              

                w = w + 1

        train_data = np.append(train_data1, train_data2, axis = 1)
        
        np.random.shuffle(train_data)

        validation_data1 = np.zeros((validation_size, (num_items + 1)))
        validation_data2 = np.zeros((validation_size, num_items))
        train_data12 = np.zeros((train_size, (num_items + 1)))
        train_data22 = np.zeros((train_size, num_items))
        
        prob_val = np.zeros((validation_size, 1))


        for i in range(train_size):
            train_data12[i][:] = train_data[i][0:num_items+1]
            train_data22[i][:] = train_data[i][num_items+1:(2*(num_items)+1)]

       
        for i in range(train_size, train_size + validation_size):
            validation_data1[i-train_size][:] = train_data[i][0:num_items+1]
            validation_data2[i-train_size][:] = train_data[i][num_items+1:(2*(num_items)+1)]
            prob_val[i-train_size][0] = prob_train[i][0]
        

           
        return prob, r, train_data1, train_data2, train_data12, train_data22, validation_data1, validation_data2, test_data1, test_data2, prob_test, prob_val, pairs_train0, pairs_train1

    
    def _generate_ranking_data(self, U, num_items):

        total_pair = int(comb(num_items,2))

        features = np.copy(U)

        
        score = np.zeros(num_items)
        keys = np.zeros(num_items)

        prob_data = np.zeros((num_items,(num_items+1)))
        prob_data_test = np.zeros((num_items,num_items))
        rank_data1 = np.zeros((num_items-1,num_items))
        rank_data2 = np.zeros((num_items-1,num_items))
        rank_data12 = np.zeros((int(comb(num_items,2)),num_items))
        rank_data22 = np.zeros((int(comb(num_items,2)),num_items))

        w = 0
               
        i = 0
        for j in range(i+1,num_items):
            rank_data1[w][i] = 1
            rank_data2[w][j] = 1
            
            w = w + 1

        w = 0
                
        for i in range(num_items-1):
            for j in range(i+1, num_items):
                rank_data12[w][i] = 1
                rank_data22[w][j] = 1
            
                w = w + 1

        #prob_embedding = np.zeros((num_items, num_items))

        #print(U)
        
        for i in range(num_items-1):
            for j in range(i+1, num_items):
        
                diff = np.abs(features[:,i] - features[:,j])
                diff2 = np.amax(diff)
                ind = np.argmax(diff)
                diff3 = (features[ind,i] - features[ind,j])
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
            score[i] = prob_data[i][num_items]
            keys[i] = i


        

        d = dict(zip(keys, score))
        #print(d)

        items_ranked_original = []
        score_sorted_original = []

        for k in sorted(d, key=d.get, reverse=True):
            items_ranked_original.append(k)
            score_sorted_original.append(d[k])

        #print("items_ranked_original = ", items_ranked_original)

        #print(rank_data12)
              

        return prob_data_test, items_ranked_original, rank_data1, rank_data2, rank_data12, rank_data22, prob_data, score          

                        

        
                        


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

        
