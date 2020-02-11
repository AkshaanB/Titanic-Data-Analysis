#!/usr/bin/env python
# coding: utf-8

# **Collecting and Importing the data**

# In[10]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[3]:


titanic_url = "D:\My Work\Otherthan Syllabus\Data Science\Projects\Titanic Dataset\TitanicData.csv"


# In[111]:


df = pd.read_csv(titanic_url)


# In[112]:


df


# In[48]:


print("The no of passengers: ",len(df))


# **Analyzing the data**

# In[14]:


#Compoaring the passengers who survived and who did not
sns.countplot(x="Survived",data=df)
#So, 0 represents the non-survivors and 1 represents the survivors


# In[15]:


# Of the survivors how many were men and how many were women
sns.countplot("Survived",hue="Sex",data=df)


# In[17]:


# Of the survivors which class passengers survived the most
sns.countplot("Survived",hue="Pclass",data=df)
# So 3rd class passengers are the most non-survivors


# In[18]:


# different ages of passengers
df['Age'].plot.hist()


# **Data Wrangling**

# In[113]:


df.isnull().sum()


# In[114]:


#dropping the Cabin cloumn
df.drop("Cabin",axis=1,inplace=True)


# In[115]:


df


# In[116]:


df.isnull().sum()


# In[117]:


df.dropna(subset=["Embarked"],axis=0,inplace=True)


# In[118]:


df.isnull().sum()


# In[119]:


# replacing the null values of age with mean age of the cloumn
mean_age = df['Age'].mean()
mean_age


# In[120]:


df['Age'] = df['Age'].replace(np.nan,mean_age)
df['Age']


# In[121]:


df.isnull().sum()


# In[122]:


#rounding off the age
df['Age'].round()


# In[123]:


df['Age'] = df['Age'].astype('int')


# In[124]:


df.info()


# In[125]:


df.dtypes


# In[126]:


df.isnull().sum()


# In[127]:


df


# In[128]:


#converting sex, pclass and embarked to dummy variables
sex = pd.get_dummies(df['Sex'],drop_first=True)
sex.columns = ['Male']
sex


# In[81]:


#if Male is 1 then male else if it is 0 then female


# In[129]:


ebmarked = pd.get_dummies(df['Embarked'],drop_first = True)
ebmarked


# In[85]:


#if Q is 1 then its Q else if S is 1 then S else if both Q and S is 0 then it is C


# In[130]:


pclass = pd.get_dummies(df['Pclass'],drop_first = True)
pclass.columns = ['2nd class','3rd class']
pclass


# In[88]:


#if 2nd class is 1 then its 2nd class else if 3rd class is 1 then 3rd class else if both 2nd class and 3rd class is 0 then it is 1st class


# In[131]:


df = pd.concat([df,sex,ebmarked,pclass],axis=1)
df


# In[132]:


df = df.drop(['PassengerId','Pclass','Name','Sex','Ticket','Embarked'],axis=1)


# In[133]:


df


# **Training the data**

# In[135]:


x = df.drop(['Survived'],axis=1)
y = df['Survived']


# In[138]:


from sklearn.model_selection import train_test_split


# In[139]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[140]:


from sklearn.linear_model import LogisticRegression


# In[141]:


log_model = LogisticRegression()


# In[142]:


log_model.fit(x_train,y_train)


# In[143]:


prediction = log_model.predict(x_test)


# In[144]:


from sklearn.metrics import classification_report


# In[145]:


classification_report(y_test,prediction)


# In[146]:


from sklearn.metrics import confusion_matrix


# In[147]:


confusion_matrix(y_test,prediction)


# In[148]:


from sklearn.metrics import accuracy_score


# In[150]:


accuracy = accuracy_score(y_test,prediction)*100


# In[151]:


print(accuracy.round())


# In[ ]:


#So the model is 84% accurate

