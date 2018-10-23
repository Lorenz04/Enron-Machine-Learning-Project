
# coding: utf-8


# The enron dataset contains email (emails written) and financial data (salary, bonus etc.) from enron employees. 
# 
# The goal of this project is to find out what features identify a person of interest (POI). For this purpose we can apply machine learning to train a model on a dataset with these features. Based on these features the algorithmn can tell us if a specific person is a POI or not.


#!/usr/bin/python

import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from feature_format import featureFormat, targetFeatureSplit

features_list = ['poi', 'salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments',
                 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 
                 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 
                 'total_stock_value', 'to_messages', 'from_messages', 'from_this_person_to_poi', 
                 'from_poi_to_this_person', 'shared_receipt_with_poi', 'fraction_from_poi_email', 'fraction_to_poi_email']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


# In[84]:


# dict to dataframe
df = pd.DataFrame.from_dict(data_dict, orient='index')
df.replace('NaN', np.nan, inplace = True)


# In[85]:


df.info()


# In[86]:


df.head()


# In[87]:


### number of poi
len(df[df['poi']])


# ### Getting to know the data
# 
# We have in total 146 observations and 21 variables in our dataset - 6 email features, 14 financial features and 1 POI label - and they are divided between 18 POI’s and 128 non-POI’s.
# 
# Now that we got a feel for the dataset, lets visualize the data and check for outliers, especially when plotting salary and bonus.

# In[88]:


features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


# In[89]:


df['salary'].idxmax()


# ### Find and Remove Outliers
# 
# We clearly see an outlier, which seems to be the "total". Lets remove the "total". Also remove NaN's.

# In[90]:


### remove outliers before proceeding further
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

from pprint import pprint

outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key,int(val)))

pprint(sorted(outliers,key=lambda x:x[1],reverse=True)[:2])


# After removing the outlier, the plot shows a better picture. We still have some data points with very high x and y values, but this seems ok for C-level positions and possibly POIs.

# ### Creating New Features
# 
# It is quite possible that POI have relatively high bonus and salary numbers. However, lets create some new features. POIs possible wrote a lot of emails back and forth. Moreover, other potential POIs wrote emails to actual POI and also received emails from POIs. Therefore, we define the features fraction_to_poi_email and fraction_from_poi_email. Seeing how frequently a person communicates with a POI compared to non-POIs is potentially more useful than a raw count. Based on these ideas, the following features are created.

# In[91]:


### create new features
### new features: fraction_to_poi_email, fraction_from_poi_email

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1

### store to my_dataset for easy export below
my_dataset = data_dict

### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)


# In[92]:


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# ### ML starts here
# 
# In this section we will apply machine learning. Specifically, I will first split my data into training and testing data. In machine learning this is common practics to avoid overfitting. 
# 
# When using the k-fold cross validation method, the training set is split into k smaller sets. A model is trained using k-1 of the folds as training data; the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a performance measure such as accuracy).The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop.
# 
# Now that we have split the data, can can apply an algorithmn. I first applied the decision tree classifier. Decision trees ask multiple linear questions, one after another. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

# In[93]:


### ML goes here!

### deploying feature selection
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]
    


# ### Feature Scaling
# I have used decision tree as my final algorithm. Algorithms like decision tree and linear regression don't require feature scaling, whereas Support Vector Machines (SVM) and k-means clustering does.
# 
# SVM and k-means clustering calculate Euclidean distance between points. If one of the features has a large range, the distance will be governed by this particular feature. These classifiers are affine transformation variant.
# 
# In case of linear regression, there is a coefficient for each feature. If a feature has large ranges that do not effect the label, the regression algorithm will make the corresponding coefficients small.

# ### Feature Selection
# 
# Having more features does not necessarily mean more Information. We want to have the minimum number of features than can capture the most trends and patterns in our data. We want to get rid of features that do not give us any extra information. The  algorithm is going to be as good as the features we put into it.
# 
# I used SelectKBest on features and selected the 10 best features.

# In[94]:


from sklearn.feature_selection import SelectKBest, f_classif


# Perform feature selection
# chose 10 best features

selector = SelectKBest(f_classif, k=10)
s = selector.fit_transform(features, labels)

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

print (scores)


# In[105]:


### try Naive Bayes for prediction

t0 = time()

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)

### Accuracy NB
print (accuracy)

### NB algorithm time
print (round(time()-t0, 3), "s")

# function for calculation ratio of true positives
# out of all positives (true + false)
print ('precision = ', precision_score(labels_test,pred))

# function for calculation ratio of true positives
# out of true positives and false negatives
print ('recall = ', recall_score(labels_test,pred))

# function for calculation of f1 score

print ('f1 = ', f1_score(labels_test,pred))


# In[109]:


from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)

### Accuracy before tuning
print (score)

### Decision tree algorithm time
print (round(time()-t0, 3), "s")

# function for calculation ratio of true positives
# out of all positives (true + false)
print ('precision = ', precision_score(labels_test,pred))

# function for calculation ratio of true positives
# out of true positives and false negatives
print ('recall = ', recall_score(labels_test,pred))

# function for calculation of f1 score

print ('f1 = ', f1_score(labels_test,pred))


# In[107]:


importances = clf.feature_importances_
import numpy as np#
indices = np.argsort(importances)[::-1]

for i in range(10):
    print ("{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]]))


# I first tried a naive bayes classifier. The NB model shows an accuracy score of 0.75, a recall score of 0.6 and a precision score of 0.23. Additionally, I got an f1 score of 0.33.
# 
# In the following I used a decision tree classifier and ranked the 10 features according to their importance.
# 
# I first started with the default parameters and got an accuracy score of 85.4% after 0.003s. I got a recall score of 0.6 and a precision score of 0.23. Additionally, I got an f1 score of 0.33.
# 
# I ranked the feature importance. Salary seems to be the feature of highest importance when identifying POIs (followed by bonus and long term incentive). 
# 
# Since my decision tree classifier performed a little better, I continued with this one and tuned it.

# ### Tuning the Algo
# A very important part of machine learning is to adjust the parameters of an algorithm in order to maximize the evaluation metrics and subsequently optimize its performance. If the parameters are not properly tuned, the algorithm can underfit or overfit the data, and thus producing suboptimal results.
# 
# In order to tune my decision tree algorithm, I used the GridSearchCV tool provided in scikit learn. It searches for the best parameters between the ones specified in an array of possibilities. The parameters are chosen in order to optimize a chosen scoring function, in my case the f1 score.

# In[113]:


parameters = {'criterion':('gini', 'entropy')
                }
DT = DecisionTreeClassifier(random_state = 10)
clf = GridSearchCV(DT, parameters, scoring = 'f1')
clf= clf.fit(features_train, labels_train)
clf = clf.best_estimator_

estimators = [('scaler', MinMaxScaler()),
            ('reduce_dim', PCA()), 
            ('clf', clf)]


# In[114]:


clf = Pipeline(estimators)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)
print ("Accuracy: ", accuracy)
target_names = ['non_poi', 'poi']
print (classification_report(y_true = labels_test, y_pred =pred, target_names = target_names))


# ### Results
# I decided to do some tuning to my classifier to achieve better precision, recall and f1 score.
# 
# 
# My model achieved an overall accuracy score of 83.3%, a precision score of 86% and a recall score of 83%. Additionally, the model has an f1 score of 85%. The classifier beats the .3 both times.
# 
# 
# ### Interpretation
# Precision measures the ratio of true positives (meaning a real POI is also predicted to be a POI) out of true positives plus false positives.
# A high precision states that nearly every time a POI shows up in my test set, I am able to identify him or her.
# 
# And recall measures the ratio of true positives in relation to true positives and false negatives.
# A high recall rate states that I am good at NOT falsely predicting POIs.
# 
# F1 is a way of balance precision and recall, and is given by the following formula:
# 
# $$F1 = 2 * (precision * recall) / (precision + recall)$$
# 
# A good F1 score means both my false positives and false negatives are low, I can identify my POI's reliably and accurately. If my classifier flags a POI then the person is almost certainly a POI, and if the classifier does not flag someone, then they are almost certainly not a POI.

# In[115]:


### dump your classifier, dataset and features_list so
### anyone can run/check your results

pickle.dump(clf, open("my_classifier.pkl", "wb") )
pickle.dump(data_dict, open("my_dataset.pkl", "wb") )
pickle.dump(features_list, open("my_feature_list.pkl", "wb") )

