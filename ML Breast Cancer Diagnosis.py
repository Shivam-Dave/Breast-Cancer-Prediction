#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv(r'C:\Users\acer\Downloads\breast-cancer-dataset ML .csv')
display(df.head())


# In[2]:


missing_values = df.isnull().sum()

print("\nMissing values:")
print(missing_values)


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


from sklearn.model_selection import train_test_split

# Separate the features (X) and the target variable (y)
X = df[['Age', 'Menopause', 'Tumor Size (cm)', 'Inv-Nodes', 'Metastasis']]
y = df['Diagnosis Result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# In[7]:


from sklearn.ensemble import RandomForestClassifier

# Step 2: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Model Selection
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Model Training
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_rep}')


# In[8]:


from sklearn.linear_model import LogisticRegression


# Step 3: Model Selection
model = LogisticRegression()

# Step 4: Model Training
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_rep}')


# In[9]:


from sklearn.svm import SVC

# Step 3: Model Selection
model = SVC(kernel='linear')

# Step 4: Model Training
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_rep}')


# In[10]:


from sklearn.tree import DecisionTreeClassifier

# Step 3: Model Selection
model =DecisionTreeClassifier()

# Step 4: Model Training
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_rep}')


# In[11]:


from sklearn.naive_bayes import GaussianNB

# Step 3: Model Selection
model = GaussianNB()

# Step 4: Model Training
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_rep}')


# In[ ]:




