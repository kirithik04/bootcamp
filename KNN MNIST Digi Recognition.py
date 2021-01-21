#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[52]:


dfx=pd.read_csv('xdata.csv')
dfy=pd.read_csv('ydata.csv')


# In[53]:


X=dfx.values
Y=dfx.values

X=X
print(X)
print(X.shape)
print(Y.shape)


# In[54]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[55]:


query_x=np.array([2,3])
plt.scatter(X[:,0],X[:,1],c=Y)
plt.scatter(query_x[0],query_x[1],color='red')
plt.show()


# In[56]:


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def knn(X,Y,queryPoint,k=5):
    vals=[]
    m=X.shape[0]
    for i in range(m):
        d=dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
        vals=sorted(vals)
        vals=vals[:k]
        
        vals=np.array(vals)
        
        #print(vals)
        
        new_vals=np.unique(vals[:,1],return_counts=True)
        print(new_vals)
        
        index=new_vals[1].argmax()
        pred=new_vals[0][index]
        
        return pred


# In[57]:


knn(X,Y,query_x)


# # MNIST DATASETS

# In[58]:


df=pd.read_csv('train.csv')
print(df.shape)


# In[59]:


print(df.columns)


# In[60]:


df.head()


# In[61]:


#create numpy array
data=df.values
print(data.shape)
print(type(data))


# In[62]:


X=data[:,1:]
Y=data[:,0]
print(X.shape,Y.shape)


# In[63]:


split=int(0.8*X.shape[0])
print(split)


# In[64]:


X_train=X[:split,:]
Y_train=Y[:split]
X_test=X[split:,:]
Y_test=Y[split:]

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:





# In[65]:


def drawImg(sample):
    img=sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.show()


# In[66]:


drawImg(X_train[3])
print(Y_train[3])


# In[67]:


#making prediction


# In[68]:


pred=knn(X_train,Y_train,X_test[0])
print(pred)


# In[69]:


drawImg(X_test[7])
print(Y_test[7])


# In[ ]:





# In[ ]:




