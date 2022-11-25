#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import copy
import pylab as py
import datetime as dt
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV


##importing algorithms

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import xgboost as xgb

import seaborn as sns

import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning) 
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = RuntimeWarning) 
wrn.filterwarnings('ignore', category = UserWarning) 


# In[2]:


df=pd.read_csv("Sales.csv")


# In[3]:


df.head()

# In[5]:


df.info()


# In[6]:


df.isnull()


# In[7]:


df.isnull().sum().sum()


# In[8]:


data = df[['CustomerID','TOTAL_ORDERS','REVENUE','FIRST_ORDER_DATE','LATEST_ORDER_DATE','AVGDAYSBETWEENORDERS']]


# In[9]:


data.head()


# In[10]:


data.max()


# In[11]:


Latest_Date = dt.datetime(2021,10,24)
RFM_value = data.groupby('CustomerID').agg({'LATEST_ORDER_DATE': lambda x: (Latest_Date - pd.to_datetime(x.max())).days,
                                                'AVGDAYSBETWEENORDERS': lambda x: int(x), 'REVENUE': lambda x: int(x)})
RFM_value['LATEST_ORDER_DATE'] = RFM_value['LATEST_ORDER_DATE'].astype(int)
RFM_value.rename(columns={'LATEST_ORDER_DATE': 'Recency', 
                         'AVGDAYSBETWEENORDERS': 'Frequency', 
                         'REVENUE': 'Monetary'}, inplace=True)
RFM_value.reset_index()


# In[12]:


#split into groups 
#0-25%
#25%-50%
#50%-75%
#75%-100%


quantiles = RFM_value.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


def RScoring(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4

def FnMScoring(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# In[13]:


RFM_value['R'] = RFM_value['Recency'].apply(RScoring, args=('Recency',quantiles))
RFM_value['F'] = RFM_value['Frequency'].apply(FnMScoring, args=('Frequency',quantiles))
RFM_value['M'] = RFM_value['Monetary'].apply(FnMScoring, args=('Monetary',quantiles))


# In[14]:



RFM_value['RFMGroup'] = RFM_value.R.map(str) + RFM_value.F.map(str) + RFM_value.M.map(str)

RFM_value['RFMvalue'] = RFM_value[['R', 'F', 'M']].sum(axis = 1)



# In[15]:



def handel_badge(value):
    if(value>=10):
        return 'Bronze'
    elif(value>=8 and value<10):
        return 'Silver'
    elif(value>=6 and value<8):
        return 'Gold'
    else:
        return 'Platinum'

# Storing all the values in a list   
list1=RFM_value['RFMvalue'].apply(handel_badge)



# In[16]:


RFM_value['RFM_Loyalty_Level']=list1
RFM_value.reset_index()


# In[17]:


df.drop('index', axis=1, inplace=True)


# In[18]:


merged = pd.merge(df,RFM_value,on='CustomerID',how='left')
# merged.sort_values('Frequency',ascending=False)
merged['NextPurchaseDayRange'] = 1



# In[19]:


merged.loc[merged.Frequency>90, 'NextPurchaseDayRange'] = 0



# In[20]:


merged['FIRST_ORDER_DATE'] = pd.to_datetime(merged['FIRST_ORDER_DATE'])
merged['LATEST_ORDER_DATE'] = pd.to_datetime(merged['LATEST_ORDER_DATE'])
merged['RFMGroup'] = merged['RFMGroup'].astype(str).astype(int)


# In[21]:


merged = merged.drop(['FIRST_ORDER_DATE','LATEST_ORDER_DATE','Frequency','RFM_Loyalty_Level'], axis=1)


# In[22]:


X, y = merged.drop('NextPurchaseDayRange', axis=1), merged.NextPurchaseDayRange
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, shuffle=True)


# In[23]:


models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
#measure the accuracy
for name,model in models:
    kfold = KFold (n_splits=2)
    cv_result = cross_val_score (model, X_train,y_train, cv = kfold,scoring="accuracy")
    print(name, cv_result)


# In[24]:


p_grid_search = GridSearchCV(estimator = xgb.XGBClassifier(eval_metric='mlogloss'), 
                             param_grid = { 'max_depth':range(3,10,2), 'min_child_weight':range(1,5,2)}, 
                             scoring='accuracy', 
                             n_jobs=-1, 
                             #iid=False, 
                             cv=2
                            )

p_grid_search.fit(X_train, y_train)

# In[26]:


refined_xgb_model = xgb.XGBClassifier(eval_metric='logloss', 
                                      max_depth=list(p_grid_search.best_params_.values())[0]-1, 
                                      min_child_weight=list(p_grid_search.best_params_.values())[-1]+4
                                     ).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'.format(refined_xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'.format(refined_xgb_model.score(X_test[X_train.columns], y_test)))


# In[27]:


ref_xgb_pred_y = refined_xgb_model.predict(X_test)


# In[28]:


log_reg_pred_y = LogisticRegression().fit(X_train, y_train).predict(X_test)


# In[29]:


data = {'y_Actual':np.array(y_test), 'y_Predicted': ref_xgb_pred_y}
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
conf_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], 
                              rownames=['Actual'], 
                              colnames=['Predicted'])
    
sns.heatmap(conf_matrix, annot=True, fmt = "d", cmap="Spectral")
plt.show()


# In[30]:


data = {'y_Actual':np.array(y_test), 'y_Predicted': log_reg_pred_y}
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
conf_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], 
                              rownames=['Actual'], 
                              colnames=['Predicted'])
    
sns.heatmap(conf_matrix, annot=True, fmt = "d", cmap="Spectral")
plt.show()

# In[33]:
st.title("ECOMMERCE NEXT PURCHASE DAY PREDICTION")
st.subheader("Single Customer")
st.caption("Platinum - most frequent customers(orders within a month) ")
st.caption("Gold - frequent customers(orders in a month)")
st.caption("Silver - less frequent customers(orders more than a month or two)")
st.caption("Bronze - rare customers(orders very rarely) ")
number = st.number_input('Enter customer ID',1000,1000000)

pre = merged[merged['CustomerID']==number]['CustomerID'].values
if len(pre) == 0:
    st.warning('INVALID ID')
else:
    rfm = merged[merged['CustomerID']==int(number)]['RFMvalue']
    Loyalty = rfm.values
    st.write(handel_badge(Loyalty[0]))
st.subheader("Multiple Customer")
d=st.selectbox("Select type",("Platinum","Gold","Silver","Bronze"))
if d == 'Platinum':
    st.write("Displays the most frequent customers")
    st.write("Total Members",len(merged[merged['RFMvalue'] < 6 ]))
    st.write("Total Orders",np.sum(merged[merged['RFMvalue'] < 6 ]["TOTAL_ORDERS"]))
    e=merged[merged['RFMvalue'] < 6 ]
    e=e[["CustomerID","TOTAL_ORDERS","REVENUE"]]
    st.dataframe(e)
if d == 'Gold':
    st.write("Displays the frequent customers")
    st.write("Total Members",len(merged[(merged['RFMvalue'] >= 6) & (merged['RFMvalue'] <8) ]))
    st.write("Total Orders",np.sum(merged[(merged['RFMvalue'] >= 6) & (merged['RFMvalue'] <8) ]["TOTAL_ORDERS"]))
    e=merged[(merged['RFMvalue'] >=6) & (merged['RFMvalue'] <8)]
    e=e[["CustomerID","TOTAL_ORDERS","REVENUE"]]
    st.dataframe(e)
if d == 'Silver':
    st.write("Displays the less frequent customers")
    st.write("Total Members",len(merged[(merged['RFMvalue'] >= 8 ) & (merged['RFMvalue'] <10)]))
    st.write("Total Orders",len(merged[(merged['TOTAL_ORDERS'] >= 8) & (merged['TOTAL_ORDERS'] <10) ]))
    e=merged[(merged['RFMvalue'] >=8) & (merged['RFMvalue'] <10)]
    e=e[["CustomerID","TOTAL_ORDERS","REVENUE"]]
    st.dataframe(e)
    
if d == 'Bronze':
    st.write("Displays the rare customers")
    st.write("Total Members",len(merged[merged['RFMvalue'] >=10]))
    st.write("Total Orders",len(merged[merged['TOTAL_ORDERS'] >=10 ]))
    e=merged[merged['RFMvalue'] >=10]
    e=e[["CustomerID","TOTAL_ORDERS","REVENUE"]]
    st.dataframe(e)