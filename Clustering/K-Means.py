
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

path = "D:/Fall 2017/Data Mining/Project 2/new_dataset_1.txt"


data = pd.read_csv(path, sep="\t", header=None)

l=len(data.columns)
xvec = data.ix[:,2:l-1]
yvec = data.ix[:,0:1]


x = xvec.values

def randomcentroids(x, l):
    centroids = x.copy()
    #centroids
    np.random.shuffle(centroids)
    cnt = centroids[:5]
    return cnt

MAX_ITERATIONS = 10
def check_centroids(old, cnt, iters):
    if iters > MAX_ITERATIONS:
        return 1
    return np.all(old == cnt)

import math
def dist_mat(x,cnt):
    c = []
    t4 = []
    z = []
    for i in range(0,len(x)):
        arr = []
        for j in range(0,len(cnt)):
            dist = [(k-l) ** 2 for k,l in zip(x[i],cnt[j])]
            dist = math.sqrt(sum(dist))
            arr.append(dist)
       # print(arr)
        t4.append(arr)
    for i in range(0,len(t4)):
        z.append(np.argmin(t4[i]))
    return np.array(z)
    

def new_centroid(x, dist, cnt):
   # return np.array([x[dist==i].mean(axis=0) for i in range(cnt.shape([0]))])
    arr = []
    for i in range(cnt.shape[0]):
        arr.append(np.array(x[dist==i].mean(axis=0)))
    return arr

cnt = [x[2],x[4],x[8]]
iters = 0
old = None

cnt = np.array(cnt)

t4 = dist_mat(x,cnt)
t4 = np.array(t4)
while not check_centroids(old, cnt, iters):
    
    old = cnt
    iters+=1

    dm = dist_mat(x,cnt)
   # print("hi")
    cnt = new_centroid(x, dm, cnt)
    cnt = np.asarray(cnt)
    #print(old)
    #print(cnt)

carr = np.zeros((dm.shape[0],dm.shape[0]),dtype="int32")
for i in range(0,len(dm)):
    for j in range(0,len(dm)):
        if dm[i] == dm[j]:
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



from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

pca = PCA(n_components = 2)
pca.fit(x)
#print(pca.explained_variance_ratio_)
d = pca.transform(x)
#print(d)

df1 = pd.DataFrame(d)
df2 = pd.DataFrame(dm)
df1['Cluster'] = df2
df1.columns = ['X', 'Y', 'Cluster']

sns.lmplot('X', 'Y', df1, hue='Cluster', fit_reg=False, size=8
              ,scatter_kws={'alpha':0.7,'s':60})
plt.show()


