
# coding: utf-8

# In[337]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\titanic\\train.csv")

test = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\titanic\\test.csv")

train.head()


# In[338]:


a = train['Embarked'].unique()
a = list(a)
print(a)


# In[339]:


for i in range(0,len(train['Embarked'])):
    if train['Embarked'][i] in a:
        train['Embarked'][i] = str(a.index(train['Embarked'][i]))  


# In[340]:


train['Embarked'] = train['Embarked'].fillna(0)
train.head()


# In[341]:


b = train['Sex'].unique()
b = list(b)
print(b)


# In[342]:


for i in range(0,len(train['Sex'])):
    if train['Sex'][i] in b:
        train['Sex'][i] = str(b.index(train['Sex'][i]))  


# In[343]:


for i in range(0,len(train['Ticket'])):
    train['Ticket'][i] = train['Ticket'][i][:1]
train.head()


# In[344]:


c = train['Ticket'].unique()
c = list(c)
print(c)


# In[345]:


for i in range(0,len(train['Ticket'])):
    if train['Ticket'][i] in c:
        train['Ticket'][i] = str(c.index(train['Ticket'][i]))  


# In[346]:


for i in range(0,len(train['Fare'])):
    if train['Fare'][i] >= 0 and train['Fare'][i]<8:
        train['Fare'][i] = 0
    elif train['Fare'][i] >=8 and train['Fare'][i]<15:
        train['Fare'][i] = 1
    elif train['Fare'][i] >=15 and train['Fare'][i]<31:
        train['Fare'][i] = 2
    else: train['Fare'][i] = 3


# In[347]:


train = train.drop('Cabin',axis=1)


# In[348]:


for i in range(0,len(train['Age'])):
    if train['Age'][i] <=16:
        train['Age'][i] = 0
    else: train['Age'][i] = 1
train.head()


# In[349]:


train["Name"] = train["Name"].astype(str).str.split().str[1]


# In[350]:


train.head()


# In[351]:


d = train['Name'].unique()
d = list(d)
print(d)


# In[352]:


for i in range(0,len(train['Name'])):
    if train['Name'][i] in d:
        train['Name'][i] = str(d.index(train['Name'][i]))  


# In[353]:


train.head()


# In[354]:


train['Age'] = train['Age'].fillna(1.0)
train.head()


# In[355]:


features = pd.DataFrame(train['Survived'])
train = train.drop('Survived',axis=1)
features.head()


# In[356]:


train.dtypes


# In[357]:


from sklearn import preprocessing
x = train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train = pd.DataFrame(x_scaled)
train.head()


# In[358]:


for i in range(0,10):
    std = train[i].values.std()
    mean = train[i].values.mean()
    for j in range(0,len(train[i])):
        train[i][j] = (train[i][j] - mean) / std
train.head()


# In[359]:


train.dtypes


# In[360]:


features.head()


# In[363]:


test = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\titanic\\test.csv")


# In[364]:


test.head()


# In[365]:


e = test['Embarked'].unique()
e = list(e)
print(e)


# In[366]:


for i in range(0,len(test['Embarked'])):
    if test['Embarked'][i] in a:
        test['Embarked'][i] = str(a.index(test['Embarked'][i]))  


# In[367]:


test['Embarked'] = test['Embarked'].fillna(0)
test.head()


# In[368]:


for i in range(0,len(test['Sex'])):
    if test['Sex'][i] in b:
        test['Sex'][i] = str(b.index(test['Sex'][i]))  


# In[369]:


for i in range(0,len(test['Ticket'])):
    test['Ticket'][i] = test['Ticket'][i][:1]
test.head()


# In[370]:


for i in range(0,len(test['Ticket'])):
    if test['Ticket'][i] in c:
        test['Ticket'][i] = str(c.index(test['Ticket'][i]))  


# In[371]:


for i in range(0,len(test['Fare'])):
    if test['Fare'][i] >= 0 and test['Fare'][i]<8:
        test['Fare'][i] = 0
    elif test['Fare'][i] >=8 and test['Fare'][i]<15:
        test['Fare'][i] = 1
    elif test['Fare'][i] >=15 and test['Fare'][i]<31:
        test['Fare'][i] = 2
    else: test['Fare'][i] = 3


# In[372]:


test = test.drop('Cabin',axis=1)


# In[373]:


for i in range(0,len(test['Age'])):
    if test['Age'][i] <=16:
        test['Age'][i] = 0
    else: test['Age'][i] = 1
test.head()


# In[374]:


test["Name"] = test["Name"].astype(str).str.split().str[1]


# In[375]:


test.head()


# In[376]:


for i in range(0,len(test['Name'])):
    if test['Name'][i] in d:
        test['Name'][i] = str(d.index(test['Name'][i]))  


# In[377]:


test.head()


# In[379]:


test['Age'] = test['Age'].fillna(1.0)
test.head()


# In[380]:


test.dtypes


# In[381]:


test['PassengerId'] = test['PassengerId'].astype(float)
test['Pclass'] = test['Pclass'].astype(float)


# In[382]:


test.dtypes


# In[383]:


test['Name'] = test['Name'].convert_objects(convert_numeric=True)


# In[384]:


test['Sex'] = test['Sex'].astype(float)
test['SibSp'] = test['SibSp'].astype(float)
test['Parch'] = test['Parch'].astype(float)


# In[385]:


test.dtypes


# In[386]:


test['Ticket'] = test['Ticket'].convert_objects(convert_numeric=True)


# In[387]:


test.dtypes


# In[388]:


test['Ticket'] = test['Ticket'].astype(float)


# In[389]:


test['Embarked'] = test['Embarked'].convert_objects(convert_numeric=True)


# In[390]:


test['Embarked'] = test['Embarked'].astype(float)


# In[391]:


test.dtypes


# In[392]:


train.head()


# In[393]:


features.head()


# In[394]:


test['Name'] = test['Name'].fillna(1)


# In[395]:


test.columns[test.isna().any()].tolist()


# In[396]:


test.dtypes


# In[397]:


from sklearn import preprocessing
y = test.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(y)
test = pd.DataFrame(y_scaled)
test.head()


# In[398]:


for i in range(0,10):
    std = test[i].values.std()
    mean = test[i].values.mean()
    for j in range(0,len(test[i])):
        test[i][j] = (test[i][j] - mean) / std
test.head()


# In[399]:


#ML
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(train, features)


# In[400]:


a = clf.predict(test)


# In[401]:


test1 = pd.read_csv("C:\\Users\\Ayush\\Desktop\\Data\\titanic\\test.csv")


# In[402]:


result = pd.DataFrame(test1['PassengerId'])


# In[403]:


result['Survived'] = a


# In[404]:


result.head()


# In[405]:


result.to_csv("C:\\Users\\Ayush\\Desktop\\Data\\titanic\\result1.csv", index = None)

