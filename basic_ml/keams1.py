#!/usr/bin/env python
# coding: utf-8

# In[45]:


import random
import collections


# In[46]:


class KMeans:
    def __init__(self):
        self.k = 3
        


# In[47]:


    def set(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter

    def euc_dist(self, a, b):
        return sum([(ai - bi) ** 2 for ai, bi in zip(a, b)])



    # In[48]:


    def fit(self, points):
        n = len(points)
        dim = len(points[0])
        centers = [points[i] for i in random.sample(range(0, n), self.k)]
        custer = collections.defaultdict(list)


        for it in range(self.max_iter):
            # assign point to center
            for i in range(n):
                min_idx = 0
                min_dist = self.euc_dist(points[i], centers[0])
                for j in range(1, self.k):
                    cur_dist = self.euc_dist(points[i], centers[j])
                    if cur_dist < min_dist:
                        min_idx = j
                        min_dist = cur_dist
                custer[min_idx].append(points[i])

            # update center
            for i in range(self.k):
                sumv = [0] * dim
                cn = len(custer[i])
                for p in custer[i]:
                    for j in range(dim):
                        sumv[j] += p[j]
                centers[i] = list(map(lambda x: x / cn, sumv))
        return centers


# In[49]:


points = [(1, 1), (2, 2), (1, 2), (2, 1),
          (-3, -3), (-4, -4), (-3, -4), (-4, -3)
          ]
km = KMeans()
km.set(3, 3)
centers = km.fit(points)
print centers


# In[ ]:





# In[ ]:





# In[ ]:




