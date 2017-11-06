
# coding: utf-8

# In[8]:

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

# Read training data
#path = "pca_a.txt"
path = input("Enter the dataset name: ")
data = pd.read_csv(path, sep="\t", header=None)

# divide the training set into X feature vector and Y target vector
l = len(data.columns)
xvec = data.ix[:,0:l-2]
yvec = data.ix[:,l-1]
#xvec

# Computing mean of the feature vector
mn = np.mean(xvec)
# Re-shaping size to no of columns in eigen vectors
rs = xvec.shape[1]
#mn

# Computing mean centered matrix
xvec1 = xvec - mn

# Computing covariance of the feature vector
#cov = np.cov([xvec[0],xvec[1],xvec[2],xvec[3]])
cov = np.cov(xvec.T)
#cov

# Eigen values and eigen vectors are computed using np.linalg.eig method
eig_val, eig_vec = np.linalg.eig(cov)

print("Eigen Values: %s " %eig_val)
print("Eigen Vectors: %s" %eig_vec)

'''
# Asserting that eigen values and vectors obtained using this function are correct

for i in range(len(eig_val)):
    eigv = eig_vec[:,i].reshape(1,rs).T
    print("Cov: %s" %(cov.dot(eigv)))
    print("Eigen val * eigen vec: %s" %(eig_val[i] * eigv))
    np.testing.assert_array_almost_equal(cov.dot(eigv), eig_val[i] * eigv,
                                         decimal=6, err_msg='', verbose=True)
'''

# Append eigen values and corresponding eigen vectors
eig = []
for i in range(len(eig_val)):
    x = []
    x = (eig_val[i],eig_vec[:,i])
    eig.append(x)
print(eig)

# Sort the obtained eig matrix by the eigen values
eig.sort(key=lambda x:x[0], reverse=True)

# Pick top 2 eigen vectors and reshape them so that we get 2 dimensions
eig_final = np.hstack((eig[0][1].reshape(rs,1), eig[1][1].reshape(rs,1)))

# Multiply the mean centered feature vector X with the eigen 2 dimenional vector
y_transform = xvec1.dot(eig_final)

#y_transform
#type(y_transform)
#type(yvec)

yvecd = pd.DataFrame(yvec)
#yvecd
df = y_transform
#df

df['disease'] = yvecd
df.columns = ['X','Y','Disease']
#df
#yvecd

sns.lmplot('X', 'Y', df, hue='Disease', fit_reg=False, size=8
              ,scatter_kws={'alpha':0.7,'s':60})
plt.show()


#plt.scatter(df['X'], df['Y'], c=df['Disease'], s=500)
#plt.gray()

#plt.show()
'''
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(xvec)
print(pca.explained_variance_ratio_)


# In[34]:

d = pca.transform(xvec)
d


# In[35]:




# In[ ]:

sns.lmplot()

'''