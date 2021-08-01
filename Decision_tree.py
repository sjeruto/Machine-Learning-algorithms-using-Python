#!/usr/bin/env python
# coding: utf-8

# In[22]:


### Upload necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[27]:


df = pd.read_csv("/Users/sharon1/Documents/MLAA_projects/decision_tree_folder/bank.csv", sep=";")
df.head()


# In[28]:


##Explore dataset
df.columns


# In[29]:


#descriptive statistics summary
df["default"].describe()


# In[30]:


##Exploratory data analysis
df.info()


# In[35]:


sns.pairplot(df, hue = 'y')


# In[36]:


### Checking if the data is balanced(Looking at the target variable)
sns.histplot(x="y",data = df)


# In[37]:


##Check which month has the highest subscription
sns.histplot(x = "month", data=df)


# In[38]:


## Checking whether age affects subscription
sns.barplot(x="y", y="age",data=df)


# In[39]:


# Can we come up with a function that maps balance or duration of the campaign to target?
plt.plot(df["balance"], df["y"], "o")


# In[40]:


plt.plot(df["duration"], df["y"], "o")


#  Higher client balance does have a relationship with increase in subscription.

# In[41]:


sns.barplot(x="y", y="balance", data=df)


# In[42]:


## Checking for the relationship between the occupation of the client and subscription;
df.groupby(['y', 'job']).size().plot(kind='bar')


# Management level workers tend to subscribe more than others.

# In[43]:


#Determining the relationship between month and subcription; 
#Note that campaign has been conducted for some of the months, thus resulting in numbers that look higher.
df.groupby(['y', 'month']).size().plot(kind='bar')


# In[44]:


### To find out if there are any correlations between the variables;
corr = df.corr()
sns.heatmap(corr, annot=True)


# In[61]:


### Further preprocessing the data;
df.loan = df.loan.replace(["yes"], 1)
df.loan = df.loan.replace(["no"], 0)
df.housing = df.housing.replace(["yes"], 1)
df.housing = df.housing.replace(["no"], 0)
df.default = df.default.replace(["yes"], 1)
df.default = df.default.replace(["no"],0)
df.y = df.y.replace(["yes"],1)
df.y = df.y.replace(["no"],0)


# In[62]:


# Encode the rest of the variables as dummies
#select the variables to encode first
cols_to_encode = df.select_dtypes(include="object")
for col in cols_to_encode:
  df = pd.concat([df, pd.get_dummies(df[col], prefix="%s"%col)], axis=1)
  df.drop([col], axis=1, inplace=True)


# In[64]:


## Scaling ther numerical variables
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df


# In[68]:


# Split dataset into training set and test set
from sklearn.model_selection import train_test_split

#extract features and target variables
x = df.drop(columns="y")
y = df["y"]

#save the feature name and target variables
feature_names = x.columns
labels = y.unique()

#split the dataset
from sklearn.model_selection import train_test_split
X_train, test_x, y_train, test_lab = train_test_split(x,y,
                                                 test_size = 0.3,
                                                 random_state = 42)


# In[66]:


#### Building decision tree model
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth =3, random_state = 42)

# Train Decision Tree Classifer
clf= clf.fit(X_train, y_train)


# In[69]:


#Predict the response for test dataset
y_pred = clf.predict(test_x)


# In[74]:


### Evaluating the model
#Accuracy can be computed by comparing actual test set values and predicted values
# Model Accuracy, how often is the classifier correct?
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


print("Accuracy:",metrics.accuracy_score(test_lab, y_pred))


# A classification rate of 89.68% is considered a good accuracy.
# You can improve this accuracy by tuning the parameters in the Decision Tree Algorithm.

# In[75]:


print(classification_report(test_lab, y_pred))
print(confusion_matrix(test_lab, y_pred))


# In[93]:


### To visualise the tree;

#Install libraries
get_ipython().system('pip3 install graphviz')
#pip install pydotplus


# In[95]:


from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('bank.png')
Image(graph.create_png())


# Optimization of decision tree classifier performed by only pre-pruning. 
# Maximum depth of the tree can be used as a control variable for pre-pruning.
#  We can try other attribute selection measure such as entropy.

# In[100]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(test_x)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_lab, y_pred))


# Using entropy, the accuracy has remained at 89%

# In[101]:


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('bank.png')
Image(graph.create_png())


# In[ ]:




# Machine-Learning-algorithms-using-Python
