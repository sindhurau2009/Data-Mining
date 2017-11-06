
import numpy as np
import scipy as sp
import pandas as pd
import math
import sys


path = "D:/Fall 2017/Data Mining/Project 2/iyer.txt"
data = pd.read_csv(path, sep="\t", header=None)

l=len(data.columns)
xvec = data.ix[:,2:l-1]
yvec = data.ix[:,0:1]
x = xvec.as_matrix()
#print(x)

nc = 7

dist = np.zeros((x.shape[0],x.shape[0]), dtype="int32")
def distmat(x):
    for i in range(0, x.shape[0]-1):
        for j in range(0,x.shape[0]-1):
            d = [(k-l) ** 2 for k,l in zip(x[i],x[j])]
            d = math.sqrt(sum(d))
            dist[i,j] = d
    return dist


#dist = distmat(x)

from scipy.spatial import distance_matrix
dist = distance_matrix(x,x)


arr = []
for i in range(0,x.shape[0]):
    arr.append([i])
#len(arr[7])


visited = []
for i in range(0,dist.shape[0]):
    if dist.shape[0] - i <= nc:
        break
    else:
        #if i not in visited:
          #  visited.append(i)
        m1 = sys.maxsize
        s1 = 0
        s2 = 0
        for j in range(0, len(arr)):
            if j not in visited:
                for k in range(j+1,len(arr)):
                    if k not in visited:
                        m2 = sys.maxsize
                        for m in range(0,len(arr[j])):
                            for n in range(0, len(arr[k])):
                                if dist[arr[j][m],arr[k][n]] < m2:
                                    m2 = dist[arr[j][m],arr[k][n]]
                        if m2 < m1:
                            m1 = m2
                            s1 = j
                            s2 = k
        arr.append(arr[s1]+arr[s2])
        visited.append(s1)
        visited.append(s2)
                        

res = []
for i in range(len(arr)-5,len(arr)):
    res.append(arr[i])
    #print(arr[i])


#for i in range(0, len(arr)):
#    if i not in visited:
#        res.append(arr[i])

cr = np.zeros((x.shape[0]),dtype="int32")
count = 0
for i in range(0,len(res)):
    for j in range(0,len(res[i])):
       # print("i,j")
        #print(i)
        #print(j)
        cr[res[i][j]] = count
    count += 1


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
print(rand)


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
print(jcq)



from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

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

