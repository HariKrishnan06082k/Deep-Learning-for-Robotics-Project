# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 15:11:56 2023

@author: Harikrishnan
"""

import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils
from scipy import interp
from sklearn.metrics import roc_curve, auc

#set sns environ
sns.set()

#read the csv file
df = pd.read_csv("combined_object_cpoints_dataset.csv")

#check if there are any nan values in dataset
print(df.isnull().sum().sum())

#reassign the column names of contact points with x,y,z.
df.rename(columns= {'0':'x',
                    '1':'y',
                    '2':'z'}, inplace = True)

#do one hot encoding of the object class names
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[4])],remainder='passthrough')
dataframe = ct.fit_transform(df)

#convert the returned numpy obj array to pandas data frame
contact_data = pd.DataFrame(dataframe)

#assign label column name for target 
contact_data.rename(columns={7:"Labels"}, inplace = True)

#do stratified sampling 
split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
for train_index, test_index in split.split(contact_data,contact_data["Labels"]):
    strat_train_set = contact_data.loc[train_index]
    strat_test_set = contact_data.loc[test_index]

#rearranging the column orders so that target feature comes in the end.
strat_train_set = strat_train_set[[0,1,2,3,4,5,6,8,9,10,11,12,13,"Labels"]]
strat_test_set = strat_test_set[[0,1,2,3,4,5,6,8,9,10,11,12,13,"Labels"]]

#Training set
X_train = strat_train_set.iloc[:,:-1]
y_train = strat_train_set.loc[:,["Labels"]]

#Test set
X_test = strat_test_set.iloc[:,:-1]
y_test = strat_test_set.loc[:,["Labels"]]

#pairplot to see correlation
sns.pairplot(data=X_train, vars=(4,5,6))

#label encoding target 
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#fit MLP classifier
clf = MLPClassifier(solver='adam',alpha=1e-5,hidden_layer_sizes=(128,64),learning_rate="adaptive",activation="relu")
clf.fit(X_train,y_train)

#get predictions on test set
y_pred = clf.predict(X_test)

#print accuracy of test set
print("Accuracy of MLP classifier",metrics.accuracy_score(y_pred, y_test))

#plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")

#outputs the micro and macro scores
print(classification_report(y_test, y_pred))


#kfold f1-weighted
kf = KFold(shuffle=True, n_splits=5)
cv_results_kfold = cross_val_score(clf, X_test, np.argmax(y_test, axis=1), cv=kf, scoring= 'f1_weighted')

print("K-fold Cross Validation f1_weigted Results: ",cv_results_kfold)
print("K-fold Cross Validation f1_weigted Results Mean: ",cv_results_kfold.mean())

#kfold accuracy
kf = KFold(shuffle=True, n_splits=5)
cv_results_kfold = cross_val_score(clf, X_test, np.argmax(y_test, axis=1), cv=kf, scoring= 'accuracy')

print("K-fold Cross Validation f1_weigted Results: ",cv_results_kfold)
print("K-fold Cross Validation f1_weigted Results Mean: ",cv_results_kfold.mean())

# Learn to predict each class against the other
n_classes = 3 # number of class

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i], )
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#The process of drawing a roc-auc curve belonging to a specific class

plt.figure()
lw = 2 # line_width
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2]) # Drawing Curve according to 3. class 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# Process of plotting roc-auc curve belonging to all classes.

from itertools import cycle
roc_auc_scores = []
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
    roc_auc_scores.append(roc_auc[i])

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Extending the ROC Curve to Multi-Class')
plt.legend(loc="lower right")
plt.show()

