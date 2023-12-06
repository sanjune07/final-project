#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib as plt
from matplotlib.pyplot import boxplot
from sklearn.linear_model import LogisticRegression


# In[2]:


current_directory = os.getcwd()
print(current_directory)


# In[3]:


updated_dir = os.chdir(r"C:\Users\sanja")


# In[4]:


filepath = 'Hospital1.txt'
df = pd.read_csv(filepath)


# In[5]:


print(df.columns)


# In[6]:


# calculating statistics
num_readmitted_1 = np.sum(df[' Readmission'])
satisfaction_staff = np.mean(df[' StaffSatisfaction'])
satisfaction_cleanliness = np.mean(df[' CleanlinessSatisfaction'])
satisfaction_food = np.mean(df[' FoodSatisfaction'])
satisfaction_comfort = np.mean(df[' ComfortSatisfaction'])
satisfaction_communication = np.mean(df[' CommunicationSatisfaction'])


# In[7]:


#printing out descriptive statistics
print(f"Number of patients readmitted: {num_readmitted_1}.")
print(f"Average staff satisfaction: {satisfaction_staff}.")
print(f"Average cleanliness satisfaction: {satisfaction_cleanliness}.")
print(f"Average food satisfaction: {satisfaction_food}.")
print(f"Average comfort satisfaction: {satisfaction_comfort}.")
print(f"Average communication satisfaction: {satisfaction_communication}.")


# In[8]:


#calculated overall satisfaction
df['OverallSatisfaction_1'] = df[[' StaffSatisfaction', ' CleanlinessSatisfaction', 
                                ' FoodSatisfaction', ' ComfortSatisfaction',
                               ' CommunicationSatisfaction']].mean(axis=1)

boxplot(df['OverallSatisfaction_1'], showfliers=True)


# In[9]:


# logistic regression
X = df['OverallSatisfaction_1'].values.reshape(-1, 1)
Y = df[' Readmission']

log_reg = LogisticRegression().fit(X, Y)


# In[10]:


# correlation results
correlation_coefficient = log_reg.coef_[0][0]

if correlation_coefficient > 0:
    print("Logistic regression results indicated a: ")
    if correlation_coefficient > 0.5:
        print ("Moderate Correlation")
    elif correlation_coefficient > 0.7:
        print ("Strong Correlation")
    else:
        print ("Weak correlation")
else:
    print("Logistic regression results indicated: ")
    print("No correlation")
    
print(f"The correlation coefficient was {correlation_coefficient}.")


# In[11]:


# plot the data
plt.pyplot.scatter(X, Y)
plt.pyplot.xlabel('Overall Satisfaction Scores')
plt.pyplot.ylabel('Readmission')
plt.pyplot.title('Logistic Regression - Overall Satisfaction vs Readmission')
plt.pyplot.plot(X, log_reg.predict(X), color = 'blue')
plt.pyplot.xlim(2, 5)


# In[12]:


filepath = 'Hospital2.txt'
gh = pd.read_csv(filepath)


# In[13]:


print(gh.columns)


# In[14]:


# calculating statistics
num_readmitted_2 = np.sum(gh[' Readmission'])
satisfaction_staff = np.mean(gh[' StaffSatisfaction'])
satisfaction_cleanliness = np.mean(gh[' CleanlinessSatisfaction'])
satisfaction_food = np.mean(gh[' FoodSatisfaction'])
satisfaction_comfort = np.mean(gh[' ComfortSatisfaction'])
satisfaction_communication = np.mean(gh[' CommunicationSatisfaction'])


# In[15]:


#printing out descriptive statistics
print(f"Number of patients readmitted: {num_readmitted_2}.")
print(f"Average staff satisfaction: {satisfaction_staff}.")
print(f"Average cleanliness satisfaction: {satisfaction_cleanliness}.")
print(f"Average food satisfaction: {satisfaction_food}.")
print(f"Average comfort satisfaction: {satisfaction_comfort}.")
print(f"Average communication satisfaction: {satisfaction_communication}.")


# In[16]:


#calculated overall satisfaction
gh['OverallSatisfaction'] = gh[[' StaffSatisfaction', ' CleanlinessSatisfaction', 
                                ' FoodSatisfaction', ' ComfortSatisfaction',
                               ' CommunicationSatisfaction']].mean(axis=1)

boxplot(gh['OverallSatisfaction'], showfliers=True)


# In[17]:


# logistic regression
X = gh['OverallSatisfaction'].values.reshape(-1, 1)
Y = gh[' Readmission']

log_reg = LogisticRegression().fit(X, Y)


# In[18]:


# correlation results
correlation_coefficient = log_reg.coef_[0][0]

if correlation_coefficient > 0:
    print("Logistic regression results indicated a: ")
    if correlation_coefficient > 0.5:
        print ("Moderate Correlation")
    elif correlation_coefficient > 0.7:
        print ("Strong Correlation")
    else:
        print ("Weak correlation")
else:
    print("Logistic regression results indicated: ")
    print("No correlation")
    
print(f"The correlation coefficient was {correlation_coefficient}.")


# In[19]:


# plot the data
plt.pyplot.scatter(X, Y)
plt.pyplot.xlabel('Overall Satisfaction Scores')
plt.pyplot.ylabel('Readmission')
plt.pyplot.title('Logistic Regression - Overall Satisfaction vs Readmission')
plt.pyplot.plot(X, log_reg.predict(X), color = 'blue')
plt.pyplot.xlim(2, 5)


# In[21]:


# compare the logistic regression results of both hospitals (readmission rates)
if num_readmitted_1 > num_readmitted_2:
    print("The number of patients readmitted to Hospital 1 is greater than that of Hospital 2.")
elif num_readmitted_1 < num_readmitted_2:
    print("The number of patients readmitted to Hospital 1 is lower than that of Hospital 2.")
else:
    print("The number of patients readmitted to both hospitals is the same.")


# In[22]:


# compare the logistic regression results of both hospitals (patient satisfaction)
if df['OverallSatisfaction_1'] > gh['OverallSatisfaction']:
    print ("Satisfaction rates of Hospital 1 are higher than Hospital 2.")
elif df['OverallSatisfaction_1'] < gh['OverallSatisfaction']:
    print ("Satisfaction rates of Hospital 1 are lower than Hospital 2.")
else:
    print("The satisfaction rate for both hospitals is the same.")


# In[23]:


print("The average patient satisfaction of Hospital 1 was 3.6108108108108 and lower than the average patient satisfaction for Hospital 2 which was 3.6385964912281.")


# In[ ]:




