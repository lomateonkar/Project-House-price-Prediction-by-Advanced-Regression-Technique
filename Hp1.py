#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_train = pd.read_csv('../../Downloads/train (1).csv')
df_test = pd.read_csv('../../Downloads/test (1).csv')


# In[3]:


na_valid = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',  'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']


# In[4]:


na_missing = set(df_test.columns) - set(na_valid)


# In[5]:


dict1 = dict.fromkeys(na_missing, ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a','nan', 'null'])
dict2 = dict.fromkeys(na_valid, ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NULL', 'NaN', 'n/a','nan', 'null'])
dict1.update(dict2)
len(dict1)


# In[6]:


dict1['Alley']


# In[7]:


df_train = pd.read_csv('../../Downloads/train (1).csv',
                       keep_default_na=False, na_values=dict1)
df_test = pd.read_csv('../../Downloads/test (1).csv',
                      keep_default_na=False, na_values=dict1)


# In[8]:


df_train.shape


# In[9]:


df_train.head()


# In[10]:


df_train.info()


# In[11]:


df_train['Alley'].value_counts()


# In[13]:


df_train[na_valid]


# In[14]:


cat_cols = list(df_test.select_dtypes(include='object').columns)
num_cols = list(df_test.select_dtypes(exclude='object').columns)
len(cat_cols + num_cols)


# In[15]:


df_train.isna().sum()[df_train.isna().sum()>0]


# In[16]:


df_test.isna().sum()[df_test.isna().sum()>0]


# In[ ]:


from sklearn.impute import SimpleImputer


# In[18]:


cat_imputer = SimpleImputer(strategy='most_frequent')
num_imputer = SimpleImputer(strategy='median')


# In[19]:


cat_imputer.fit(df_train[cat_cols])
num_imputer.fit(df_train[num_cols])


# In[20]:


df_train[cat_cols] = cat_imputer.transform(df_train[cat_cols])
df_test[cat_cols] = cat_imputer.transform(df_test[cat_cols])

df_train[num_cols] = num_imputer.transform(df_train[num_cols])
df_test[num_cols] = num_imputer.transform(df_test[num_cols])


# In[21]:


df_train.isna().sum()[df_train.isna().sum()>0]


# In[22]:


df_test.isna().sum()[df_test.isna().sum()>0]


# # Ourlier Handling

# In[23]:


import matplotlib.pyplot as plt


# In[24]:


plt.hist(df_train['SalePrice'])


# In[25]:


df_train['SalePrice'].describe()


# In[26]:


plt.boxplot(df_train['SalePrice'])


# In[29]:


for col in cat_cols:
    print(col,':',df_train[col].nunique())


# In[32]:


# encoding on train data by one hot coding
temp = df_train['Id']
dummy = pd.get_dummies(df_train[cat_cols],prefix=cat_cols)


# In[33]:


dummy


# In[34]:


df_train.drop(cat_cols,axis=1,inplace=True)


# In[35]:


df_train.shape


# In[37]:


df_train_final = pd.concat([df_train,dummy],axis=1)


# In[38]:


dummy.shape


# In[39]:


df_train_final.shape


# In[41]:


# encoding on test data by one hot coding

dummy1 = pd.get_dummies(df_test[cat_cols],prefix=cat_cols)


# In[42]:


dummy1


# In[45]:


dummy1.shape , dummy.shape


# In[46]:


df_train = pd.read_csv('../../Downloads/train (1).csv')
df_test = pd.read_csv('../../Downloads/test (1).csv')


# In[47]:


df_train_test = pd.concat([df_train.drop('SalePrice',axis=1),df_test],axis=0)


# In[48]:


df_train_test.shape


# In[49]:


df_train_test[cat_cols] = cat_imputer.transform(df_train_test[cat_cols])
df_train_test[num_cols] = num_imputer.transform(df_train_test[num_cols])


# In[50]:


df_train_test.info()


# In[51]:


dummy2 = pd.get_dummies(df_train_test[cat_cols],prefix=cat_cols)


# In[53]:


dummy2.shape , dummy.shape ,dummy1.shape


# In[54]:


df_train_test.drop(cat_cols,axis=1,inplace=True)


# In[55]:


df_train_test.shape


# In[60]:


df_train_test_final = pd.concat([df_train_test,dummy2],axis=1)


# In[61]:


df_train_test_final.shape


# In[62]:


df_train_test_final.head()


# In[64]:


X_train = df_train_test_final.iloc[0:1460]
X_test = df_train_test_final.iloc[1460:]


# In[65]:


X_train.shape , X_test.shape


# In[66]:


y = df_train['SalePrice']


# In[67]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=100)


# In[68]:


model.fit(X_train,y)


# In[77]:


y_predict = model.predict(X_test)


# In[80]:


y_predict


# In[81]:


y_predict.shape ,df_test['Id'].shape


# In[83]:


df_sub = pd.DataFrame({'Id':df_test['Id'],'SalePrice':y_predict})


# In[84]:


df_sub


# In[85]:


df_sub.to_csv('sub3.csv',index=False)


# In[ ]:




