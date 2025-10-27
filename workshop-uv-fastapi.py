#!/usr/bin/env python
# coding: utf-8

# This is a starter notebook for an updated module 5 of ML Zoomcamp
# 
# The code is based on the modules 3 and 4. We use the same dataset: [telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

# In[1]:


import pandas as pd
import numpy as np
import sklearn


# In[2]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[3]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[4]:


data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

df = pd.read_csv(data_url)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[5]:


df


# In[6]:


y_train = df.churn


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# In[24]:


from sklearn.pipeline import make_pipeline


# In[8]:


dv = DictVectorizer()

train_dict = df[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)


# In[27]:


pipeline = make_pipeline (
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
 )


# In[23]:


# X_train[:10].todense()


# In[28]:


# train_dict[0]
train_dict  = df[numerical + categorical].to_dict(orient='records')
pipeline.fit(train_dict, y_train)


# In[30]:


customer = {
    'gender': 'male',
 'seniorcitizen': 0,
 'partner': 'no',
 'dependents': 'yes',
 'phoneservice': 'no',
 'multiplelines': 'no_phone_service',
 'internetservice': 'dsl',
 'onlinesecurity': 'no',
 'onlinebackup': 'yes',
 'deviceprotection': 'no',
 'techsupport': 'no',
 'streamingtv': 'no',
 'streamingmovies': 'no',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'electronic_check',
 'tenure': 6,
 'monthlycharges': 29.85,
 'totalcharges': 129.85}

# X= dv.transform(customer)
churn = pipeline.predict_proba(customer)[0, 1]
print('Probability of churning = ', churn)

if  churn >= 0.5:
    print('send email with promo')
else:
    print('Don\'t do anything')


# In[13]:


churn


# In[19]:


import pickle


# In[20]:


with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, pipeline), f_out)


# In[21]:


get_ipython().system('ls -lh')


# In[22]:


with open('model.bin', 'rb') as f_in:
   (dv, model) = pickle.load(f_in)


# In[32]:


# !jupyter nbconvert --to=script workshop-uv-fastapi.ipynb


# In[ ]:





# In[ ]:




