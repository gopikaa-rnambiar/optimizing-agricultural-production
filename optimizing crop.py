#!/usr/bin/env python
# coding: utf-8

# In[3]:


#manipulation
import numpy as np
import pandas as pd
#visualisation
import matplotlib.pyplot as plt
import seaborn as sns
#interaction
from ipywidgets import interact


# In[4]:


data=pd.read_csv("data.csv")


# In[5]:


print("shape",data.shape)


# In[6]:


data.head()


# In[5]:


data.tail()


# In[7]:


data.isnull().sum()


# In[9]:


data['label'].value_counts()


# In[10]:


print("average ratio of nitrogen in soil:{0:.2f}".format(data['N'].mean()))


# In[11]:


print("average ratio phosphorus in soil:{0:.2f}".format(data['P'].mean()))


# In[12]:


print("average ratiO pottasium in soil:{0:.2f}".format(data['K'].mean()))


# In[14]:


print("average temperature in celcius:{0:.2f}".format(data['temperature'].mean()))


# In[15]:


print("average ratio ph in soil:{0:.2f}".format(data['ph'].mean()))


# In[16]:


print("average ration rainfall in soil:{0:.2f}".format(data['rainfall'].mean()))


# In[9]:


@interact
def summary(crops=list(data['label'].value_counts().index)):
    x=data[data['label']==crops]
    print("--------------------------------------")
    print("statistics of nitrogen:")
    print("min nitrogen required:",x['N'].min())
    print("mean nitrogen required:",x['N'].mean())
    print("max nitrogen required:",x['N'].max())
    print("--------------------------------------")
    print("statistics of phosphorus:")
    print("min phosphorus required:",x['P'].min())
    print("mean phosphorus required:",x['P'].mean())
    print("max phosphorus required:",x['P'].max())
    print("--------------------------------------")
    print("statistics of pottasium:")
    print("min pottasium required:",x['K'].min())
    print("mean pottasium required:",x['K'].mean())
    print("maxpottasium required:",x['K'].max())
    print("--------------------------------------")
    print("statistics of temparature")
    print("min temparature required:{0:.2f}".format(x['temperature'].min()))
    print("mean temparature required:{0:.2f}".format(x['temperature'].mean()))
    print("max temparature required:{0:.2f}".format(x['temperature'].max()))


# In[16]:


@interact
def compare(condition=['N','P','K','temperature','ph','humidity','rainfall']):
    print("---------------------------------------------------------------")
    print("rice:{0:.2f}".format(data[(data['label']=='rice')][condition].mean()))
    print("blackgram:{0:.2f}".format(data[(data['label']=='blackgram')][condition].mean()))
    print("banana:{0:.2f}".format(data[(data['label']=='banana')][condition].mean()))
    print("jute:{0:.2f}".format(data[(data['label']=='jute')][condition].mean()))
    print("coconut:{0:.2f}".format(data[(data['label']=='coconut')][condition].mean()))
    print("apple:{0:.2f}".format(data[(data['label']=='apple')][condition].mean()))
    print("grapes:{0:.2f}".format(data[(data['label']=='grapes')][condition].mean()))
    print("watermelon:{0:.2f}".format(data[(data['label']=='watermelon')][condition].mean()))
    print("kidneybeans:{0:.2f}".format(data[(data['label']=='kidneybeans')][condition].mean()))
    print("mungbeans:{0:.2f}".format(data[(data['label']=='mungbeans')][condition].mean()))
    print("maize:{0:.2f}".format(data[(data['label']=='maize')][condition].mean()))
    print("cotton:{0:.2f}".format(data[(data['label']=='cotton')][condition].mean()))
    print("mango:{0:.2f}".format(data[(data['label']=='mango')][condition].mean()))
    print("lentils:{0:.2f}".format(data[(data['label']=='lentils')][condition].mean())) 


# In[18]:


@interact
def copmare(conditions=['N','P','K','ph','temperature','humidity']):
    print("crops req greater than average",conditions,'\n')
    print(data[data[conditions]>data[conditions].mean()]['label'].unique())
    print("--------------------------------")
    print("crops req less than average",conditions,'\n')
    print(data[data[conditions]<=data[conditions].mean()]['label'].unique())


# In[21]:



sns.distplot(data['rainfall'])


# In[29]:


print("crops that need high nitrogen:",data[data['N']>120]['label'].unique())
print("crops that need high phosphurus:",data[data['P']>120]['label'].unique())
print("crops that need LESS POTTASIUM:",data[data['K']<12]['label'].unique())
print("crops that need LOW nitrogen:",data[data['N']<10]['label'].unique(),'\n')

print("crops that need LOW TEMP:",data[data['temperature']<10]['label'].unique(),'\n')
print("crops that need LOW HUMIDITY:",data[data['humidity']<20]['label'].unique(),'\n')
print("crops that need high rainfall:",data[data['rainfall']>100]['label'].unique())


# In[35]:


print("summer crops:")
print(data[(data['temperature']>30)&(data['humidity']>50)]['label'].unique())
print("rainy:")
print(data[(data['rainfall']>200)&(data['humidity']>50)]['label'].unique())
print("winter:")
print(data[(data['temperature']<20)&(data['humidity']>50)]['label'].unique())


# In[36]:


from sklearn.cluster import KMeans
x=data.drop(['label'],axis=1)
x=x.values
print(x.shape)


# In[39]:


plt.rcParams['figure.figsize']=(10,4)
wcss=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method',fontsize=20)
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()


# In[42]:


#kmeans implementation
km=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means=km.fit_predict(x)

a=data['label']
y_means=pd.DataFrame(y_means)
z=pd.concat([y_means,a],axis=1)
z=z.rename(columns={0:'cluster'})

#checking cluster of each crops
print("lets check results of applying cluster ananlysis",'\n')
print("crops in 1 cluster",z[z['cluster']==0]['label'].unique())
print('--------------------------------------------------')
print("crops in 2 cluster",z[z['cluster']==1]['label'].unique())
print('--------------------------------------------------')
print("crops in 3 cluster",z[z['cluster']==2]['label'].unique())
print('--------------------------------------------------')
print("crops in 4 cluster",z[z['cluster']==3]['label'].unique())
print('--------------------------------------------------')


# In[43]:


y=data['label']
x=data.drop(['label'],axis=1)


# In[44]:


print('shape of x',x.shape)
print('shape of y',y.shape)


# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print("shape of x train",x_train.shape)
print("shape of x test",x_test.shape)
print("shape of y train",y_train.shape)
print("shape of y test",y_test.shape)


# In[48]:





# In[47]:


import sklearn


# In[51]:


#creating predictive model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)


# In[52]:


#evaluate the model prediction
from sklearn.metrics import confusion_matrix

plt.rcParams['figure.figsize']=(10,10)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,cmap='Wistia')
plt.title('Confusion Matrix For Logistic Regression',fontsize=15)
plt.show()


# In[59]:


cr=classification_report(y_test,y_pred)
print(cr)


# In[58]:


from sklearn.metrics import classification_report


# In[63]:


print(data)


# In[62]:


prediction=model.predict((np.array([[90,
                                    40,
                                    40,
                                    20,
                                    80,
                                    7,
                                    200]])))
print("predicted crop is:",prediction)


# In[64]:


prediction=model.predict((np.array([[200,
                                    30,
                                    30,
                                    30,
                                    70,
                                    7,
                                    200]])))
print("predicted crop is:",prediction)


# In[ ]:




