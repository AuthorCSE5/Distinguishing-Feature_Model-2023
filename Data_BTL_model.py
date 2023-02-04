import numpy as np
import math
from scipy.special import comb

class embedding(object):
    def __init__(self, U, num_items, l, m):
    
        #self.num_items = num_items
        #self.l = l
        
        #U = self._generate_score(num_items)
        #self.U = U
        
        prob, r, train_data1, train_data2, test_data1, test_data2, prob_test, pairs_train0, pairs_train1 = self._generate_data(U, num_items, l, m)
        self.prob = prob
        self.r = r
        self.train_data1 = train_data1
        self.test_data1 = test_data1
        self.train_data2 = train_data2
        self.test_data2 = test_data2
        self.prob_test = prob_test
        self.pairs_train0 = pairs_train0
        self.pairs_train1 = pairs_train1

        rank_data12, rank_data22, items_ranked_original, score = self._generate_ranking_data(U, num_items)
        self.rank_data12 = rank_data12
        self.rank_data22 = rank_data22
        self.items_ranked_original = items_ranked_original
        self.score = score
        
        

    '''def _generate_score(self,num_items):
        
        U = np.random.rand(num_items)

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

            s1 = features[pairs_train0[i]] #score
            s2 = features[pairs_train1[i]]
            #prob = 1/(1+np.exp(-(s1-s2)))
            prob = s1/(s1+s2)
            

            r = np.random.rand(l)

            for k in range(l):
                
                
                
                if r[k] <= prob:                                        
                   train_data1[w][num_items] = 1
                   train_data1[w][pairs_train0[i]] = 1
                   train_data2[w][pairs_train1[i]] = 1
                elif r[k] > prob and r[k] <= 1:
                   train_data1[w][num_items] = -1
                   train_data1[w][pairs_train0[i]] = 1
                   train_data2[w][pairs_train1[i]] = 1
                   
               
                w = w + 1
                    

        w = 0

        prob_test = np.zeros((int(comb(num_items,2))-m,1))

        for i in range(int(comb(num_items,2)) - m):
            
                
                s1 = features[pairs_test0[i]] #score
                s2 = features[pairs_test1[i]]
                
                diff3 = s1-s2

                #prob_test[i][0] = 1/(1+np.exp(-(diff3)))
                prob_test[i][0] = s1/(s1+s2)
                

                if (diff3 >= 0):
                   test_data1[w][pairs_test0[i]] = 1
                   test_data2[w][pairs_test1[i]] = 1
                   test_data1[w][num_items] = 1
                else:
                   test_data1[w][pairs_test0[i]] = 1
                   test_data2[w][pairs_test1[i]] = 1
                   test_data1[w][num_items] = -1
                   

                #r = np.random.rand(l)

                #for k in range(l):
                
                #r = np.random.rand()

                #if r <= prob_test[i][0]:
                #   test_data[w][num_items] = 1                   
                #elif r > prob_test[i][0] and r <= 1:
                #   test_data[w][num_items] = -1

                 
                   
                
                   

                w = w + 1
                
        return prob, r, train_data1, train_data2, test_data1, test_data2, prob_test, pairs_train0, pairs_train1

        
    
    def _generate_ranking_data(self, U, num_items):

        total_pair = int(comb(num_items,2))
        w = 0
        score = np.zeros(num_items)

        features = np.copy(U)

        rank_data12 = np.zeros((int(comb(num_items,2)),num_items))
        rank_data22 = np.zeros((int(comb(num_items,2)),num_items))

        w = 0
                
        for i in range(num_items-1):
            for j in range(i+1, num_items):
                rank_data12[w][i] = 1
                rank_data22[w][j] = 1
            
                w = w + 1
        
        '''for i in range(num_items-1):
            for j in range(i+1, num_items):
        
                s1 = U[i]
                s2 = U[j]
                
                diff3 = s1-s2
                prob = 1/(1+np.exp(-diff3))
                
                if prob >= 0.5:
                   prob_data[i][j] = prob
                   prob_data[i][num_items] = prob_data[i][num_items] + 1
                   rank_data[w][i] = 1
                   rank_data[w][j] = -1
                else:
                   prob_data[j][i] = prob
                   prob_data[j][num_items] = prob_data[j][num_items] + 1
                   rank_data[w][i] = -1
                   rank_data[w][j] = 1

                w = w + 1

        for i in range(num_items):
            score[i] = prob_data[i][num_items]

        rank = np.sort(score)'''

        items = np.zeros(num_items)
        score = features

        for i in range(num_items):
            items[i] = i          


        
        dictionary = dict(zip(items, score))


        items_ranked_original = []
        score_sorted_original = []

        for k in sorted(dictionary, key=dictionary.get, reverse=True):
            items_ranked_original.append(k)
            score_sorted_original.append(dictionary[k])

        return rank_data12, rank_data22, items_ranked_original, score          

                        

    


'''d = embedding(4, 3, 3)
print(d.U)
print(d.r)
#print(d.prob_test)
print(d.data)'''



        
