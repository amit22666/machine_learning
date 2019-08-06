#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[44]:


df  = pd.read_csv('C:/Users/JAINY/Desktop/train.csv')
df_test = pd.read_csv('C:/Users/JAINY/Desktop/test.csv')


# In[45]:


df_to_np = df.values
x_test = df_test.values
x_train = df_to_np[:,1:]
y_train = df_to_np[:,0]
print(x_test.shape)


# In[46]:


print(x_train.shape)
print(y_train.shape)


# In[54]:


for i in range(5):
    plt.imshow(x_train[i].reshape(28,28),cmap='gray')
    plt.axis('off')
    plt.show()


# In[6]:


y_train = y_train.reshape(-1,1)
print(y_train.shape)


# In[7]:


from keras.layers import *

from keras.models import Sequential
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard


model = Sequential()


# In[8]:


# image reshaping
def preprocess_data(X,Y):
    X = X.reshape((-1,28,28,1))
    X = X/255.0
    Y = to_categorical(Y)
    return X,Y

def preprocess_testdata(X):
    X = X.reshape((-1,28,28,1))
    X = X/255.0
   # Y = to_categorical(Y)
    return X



XTrain,YTrain = preprocess_data(x_train,y_train)
# padding on train data
XTrain      = np.pad(XTrain, ((0,0),(2,2),(2,2),(0,0)), 'constant')

print(XTrain.shape,YTrain.shape)

XTest = preprocess_testdata(x_test)
# padding in test data
XTest = np.pad(XTest, ((0,0),(2,2),(2,2),(0,0)), 'constant')
print(XTest.shape)


# In[9]:


model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,1)))
model.add(MaxPool2D((2,2),strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D((2,2),strides=2))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())

model.add(layers.Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))


model.add(layers.Dense(units=10, activation = 'softmax'))


# In[ ]:





# In[10]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[11]:


his = model.fit(XTrain, YTrain, epochs=44, batch_size=128)


# In[18]:


results = model.predict_classes(XTest, batch_size=128)


# In[19]:


#file = np.savetxt('C:/Users/JAINY/Desktop/ans2.csv',results,delimiter=","'w+',header='Id')


# In[22]:





# In[ ]:





# In[ ]:





# In[ ]:




