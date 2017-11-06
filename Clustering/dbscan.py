# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 00:59:37 2017

@author: sindh
"""

import numpy as np
import scipy as sp
import pandas as pd
import math

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt


path = "iyer.txt"
data = pd.read_csv(path, sep="\t", header=None)

l=len(data.columns)
xvec = data.ix[:,2:l-1]
yvec = data.ix[:,0:1]
x = xvec.as_matrix()
visited = []
C = []
noise = []

def DBSCAN(x, eps, minpts):
   # c = []
    
    #noise = []
    for i in range(0, len(x)):
        if i not in visited:
            visited.append(i)
            nbpts = regionQuery(i, eps)
            if len(nbpts) < minpts:
                # append the point i as noise (outlier)
                noise.append(i)
            #print("NBPTS:")
            #print(nbpts)
            else:
                # append point i to next cluster
                c1 = []
                expandcluster(i, nbpts, c1, eps, minpts)

def expandcluster(i, nbpts, c1, eps, minpts):
    c1.append(i)
    #print("Expand Cluster")
    for j in nbpts:
     #   print("nbpts:")
        
     #   print(len(nbpts))
        if j not in visited:
            visited.append(j)
            #print("j")
            #print(j)
            nbpts1 = regionQuery(j, eps)
           
            if len(nbpts1) >= minpts:
      #          print("nbpts1")
      #          print(len(nbpts1))
               # nbpts.append(nbpts1)
                for l in nbpts1:
                    if l not in visited:
                        nbpts.append(l)
                        
        if j not in (k for k in c1):
                c1.append(j)
    C.append(c1)
                
def regionQuery(p, eps):
    arr = []
    for i in range(0,len(x)):
        dist = [(j-k) ** 2 for j,k in zip(x[p],x[i])]
        dist = math.sqrt(sum(dist))
        if dist <= eps:
            arr.append(i)
    return arr
    #print(dist)

DBSCAN(x,3.5,4)


carr = np.zeros((x.shape[0],x.shape[0]),dtype="int32")
cr = np.zeros(x.shape[0])
count = 1
for i in range(0,len(C)):
    for j in range(0,len(C[i])):
        cr[C[i][j]] = count
    count = count+1
    
carr = np.zeros((cr.shape[0],cr.shape[0]),dtype="int32")

for i in range(0,len(cr)):
    for j in range(0,len(cr)):
        if cr[i] == cr[j]:
            carr[i,j] = 1
        else:
            carr[i,j] = 0
    
y = yvec[1]
garr = np.zeros((y.shape[0],y.shape[0]),dtype="int32")
for i in range(0,len(y)):
    for j in range(0,len(y)):
        if y[i] == y[j]:
            garr[i,j] = 1
        else:
            garr[i,j] = 0


# Jaccard Coefficient

m11 = 0
m01 = 0
m10 = 0
for i in range(0, len(garr)):
    for j in range(0, len(garr)):
        if garr[i,j] == 1 and carr[i,j] == 1:
            m11 += 1
        elif garr[i,j] == 0 and carr[i,j] == 1:
            m01 += 1
        elif garr[i,j] == 1 and carr[i,j] == 0:
            m10 += 1
jcq = m11 / (m11 + m01 + m10)
print("Jaccard Coefficient:")
print(jcq)


# Rand Coefficient

m11 = 0
m01 = 0
m10 = 0
m00 = 0
for i in range(0, len(garr)):
    for j in range(0, len(garr)):
        if garr[i,j] == 1 and carr[i,j] == 1:
            m11 += 1
        elif garr[i,j] == 0 and carr[i,j] == 1:
            m01 += 1
        elif garr[i,j] == 1 and carr[i,j] == 0:
            m10 += 1
        else:
            m00 += 1
rand = (m11 + m00) / (m11 + m01 + m10 + m00)
print("Rand Coefficient:")
print(rand)


# Use PCA to reduce dimensions and plot the graphs

pca = PCA(n_components = 2)
pca.fit(x)
#print(pca.explained_variance_ratio_)
d = pca.transform(x)
#print(d)

df1 = pd.DataFrame(d)
df2 = pd.DataFrame(cr)
df1['Cluster'] = df2
df1.columns = ['X', 'Y', 'Cluster']

sns.lmplot('X', 'Y', df1, hue='Cluster', fit_reg=False, size=8
              ,scatter_kws={'alpha':0.7,'s':60})
plt.show()