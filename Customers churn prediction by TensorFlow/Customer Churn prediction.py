#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head(5)


# In[32]:


data.info()


# # Data cleaning

# In[33]:


data.drop('customerID', axis=1, inplace=True)


# In[34]:


data.TotalCharges.values


# In[35]:


data= data[data.TotalCharges !=' ']
data.shape


# In[36]:


data['TotalCharges'] = pd.to_numeric(data.TotalCharges, errors='coerce')
#coerce used to ignore the errors inside the attribute


# In[39]:


data.dtypes


# In[40]:


data.head(1)


# In[41]:


tenure_churn_no=data[data.Churn=='No'].tenure
tenure_churn_yes=data[data.Churn=='Yes'].tenure

plt.hist([tenure_churn_no,tenure_churn_yes], color=['green','red'],label=['Churn=No','Churn=Yes'])
plt.title('Customer churn visualization based on tenure')
plt.xlabel('Tenure')
plt.ylabel('Number of customer')
plt.legend()


# In[42]:


tenure_churn_no=data[data.Churn=='No'].MonthlyCharges
tenure_churn_yes=data[data.Churn=='Yes'].MonthlyCharges

plt.hist([tenure_churn_no,tenure_churn_yes], color=['green','red'],label=['Churn=No','Churn=Yes'])
plt.title('Customer churn visualization based on Monthy Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Number of customer')
plt.legend()


# In[43]:


def obj_column_values(data):
    for column in data:
        if data[column].dtypes =='object':
            print(f'{column} : {data[column].unique()}')


# In[44]:


obj_column_values(data)


# In[45]:


#Want to replace 'No....' with 'No'
data.replace('No phone service','No', inplace=True)
data.replace('No internet service','No', inplace=True)


# In[46]:


# replace 'Yes' and 'No' with 1 & 0 respectively
yes_no_columns =['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
                 'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    data[col].replace({'Yes':1, 'No':0},inplace=True)


# In[47]:


obj_column_values(data)


# In[48]:


data['gender'].replace({'Female':0,'Male':1},inplace=True)


# In[49]:


data['gender'].unique()


# In[50]:


df=pd.get_dummies(data=data,columns=['InternetService','Contract','PaymentMethod'])
df.columns


# In[51]:


df.sample(3)


# In[52]:


#scaling columns for standard calcuation
columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[53]:


df[columns_to_scale]=scaler.fit_transform(df[columns_to_scale])
df.sample(2)


# In[54]:


X =df.drop(columns='Churn', axis=1)
y =df['Churn']


# In[55]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[56]:


print(X_train.shape)
print(X_test.shape)


# In[57]:


X_train[:5]


# In[58]:


import tensorflow as tf
from tensorflow import keras


# In[63]:


model = keras.Sequential([
        tf.keras.layers.Dense(25, input_shape=(26,), activation='relu'),
        tf.keras.layers.Dense(15, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)


# In[64]:


model.evaluate(X_test, y_test)


# In[65]:


y_hat=model.predict(X_test)
y_hat[:5]


# In[69]:


y_test[:10]


# In[67]:


y_pred = []
for element in y_hat:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[70]:


y_pred[:10]


# In[71]:


from sklearn.metrics import confusion_matrix, classification_report
print(classification_report(y_test,y_pred))


# In[75]:


cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:


Thank You

