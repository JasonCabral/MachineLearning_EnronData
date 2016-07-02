#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 
'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 
# You will need to use more features
## Removed loan_advances because there are very few data points, all of which are removed from
## my adjusted payments variable


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Cleaning entries with shift errors 
f_keys = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments',
'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 'exercised_stock_options', 
'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']
bhatnagar_values = ['NaN','NaN','NaN','NaN','NaN','NaN','NaN',137864,'NaN',137864,15456290,2604490,-2604490,15456290]
belfer_values = ['NaN','NaN','NaN',-102500,'NaN','NaN','NaN',3285,102500,3285,'NaN',44093,-44093,'NaN']
for i in range(0,len(f_keys)):
	data_dict['BHATNAGAR SANJAY'][f_keys[i]]=bhatnagar_values[i]
	data_dict['BELFER ROBERT'][f_keys[i]]=belfer_values[i]

pois = 0
for i in data_dict:
	if data_dict[i]['poi']==1:
		pois += 1
print "Number of POIs in Initial Data:", pois

### Task 2: Remove outliers

### Remove 'TOTAL' entry
data_dict.pop('TOTAL')

### Check 'poi' for unexpected values
# for i in data_dict:
# 	print data_dict[i]['poi']

### Outlier Checker
# x_points_list = []
# y_points_list = []
# for i in data_dict:
# 	x_points_list.append(data_dict[i]["from_poi_to_this_person"])
# 	y_points_list.append(data_dict[i]["shared_receipt_with_poi"])
# for i in range(0, len(x_points_list)):
# 	if x_points_list[i] == 'NaN':
# 		x_points_list[i] = 0.0
# 	if y_points_list[i] == 'NaN':
# 		y_points_list[i] = 0.0

# plt.scatter(x_points_list, y_points_list)
# plt.axis([min(x_points_list), max(x_points_list), min(y_points_list), max(y_points_list)])
# plt.show()

### Check high expense outliers 
# for i in data_dict:
# 	if data_dict[i]['expenses'] >= 200000 and data_dict[i]['expenses'] != 'NaN':
# 		print i, data_dict[i]

### Check high 'other' outliers
# for i in data_dict:
# 	if data_dict[i]['other'] >= 6000000 and data_dict[i]['other'] != 'NaN':
# 		print i, data_dict[i]

### Check high 'long_term_incentive' outliers
# for i in data_dict:
# 	if data_dict[i]['long_term_incentive'] >= 3000000 and data_dict[i]['long_term_incentive'] != 'NaN':
# 		print i, data_dict[i]

### Check high 'restricted_stock' outliers
# for i in data_dict:
# 	if data_dict[i]['restricted_stock'] >= 6000000 and data_dict[i]['restricted_stock'] != 'NaN':
# 		print i, data_dict[i]

### Count number of entries with no email
# nan_mails = 0
# for i in data_dict:
# 	if data_dict[i]['to_messages'] == 'NaN' and data_dict[i]['from_messages'] == 'NaN' and \
# 	data_dict[i]['from_poi_to_this_person'] == 'NaN' and data_dict[i]['from_this_person_to_poi'] == 'NaN':
# 		nan_mails += 1
# print nan_mails

### Check high email outliers
# for i in data_dict:
# 	if data_dict[i]['from_messages'] >= 10000 and data_dict[i]['from_messages'] != 'NaN':
# 		print i, data_dict[i]
# for i in data_dict:
# 	if data_dict[i]['to_messages'] >= 10000 and data_dict[i]['from_messages'] != 'NaN':
# 		print i, data_dict[i]
# for i in data_dict:
# 	if data_dict[i]['from_poi_to_this_person'] >= 400 and data_dict[i]['from_messages'] != 'NaN':
# 		print i, data_dict[i]
# for i in data_dict:
# 	if data_dict[i]['from_this_person_to_poi'] >= 400 and data_dict[i]['from_messages'] != 'NaN':
# 		print i, data_dict[i]

### Potential removal list:
#[URQUHART JOHN A (no email, high expenses), MARTIN AMANDA K (very high long term incentive, but average salary),
# KAMINSKI WINCENTY J(from messages is high enough to be an outlier),
# SHAPIRO RICHARD S(to messages high enough to be outlier), KEAN STEVEN J(to messages high enough to be outlier), 
# HUMPHREY GENE E(all outgoing emails went to POIs, but only 17 outgoing emails)]
# 
### Removed one by one to analyze effect on evaluation metrics
# data_dict.pop('URQUHART JOHN A')
# data_dict.pop('MARTIN AMANDA K')
# data_dict.pop('KAMINSKI WINCENTY J')
# data_dict.pop('SHAPIRO RICHARD S')
data_dict.pop('KEAN STEVEN J')
# data_dict.pop('HUMPHREY GENE E')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


for i in range(len(my_dataset.keys())):

	if my_dataset[str(my_dataset.keys()[i])]["loan_advances"] == "NaN":
		my_dataset[str(my_dataset.keys()[i])]["loan_advances"] = 0
	if my_dataset[str(my_dataset.keys()[i])]["expenses"] == "NaN":
		my_dataset[str(my_dataset.keys()[i])]["expenses"] = 0
	if my_dataset[str(my_dataset.keys()[i])]["deferred_income"] == "NaN":
		my_dataset[str(my_dataset.keys()[i])]["deferred_income"] = 0
	if my_dataset[str(my_dataset.keys()[i])]["deferral_payments"] == "NaN":
		my_dataset[str(my_dataset.keys()[i])]["deferral_payments"] = 0

	if my_dataset[str(my_dataset.keys()[i])]["total_payments"] == "NaN":
		my_dataset[str(my_dataset.keys()[i])]["normalized_payments"] = 0.0
		if my_dataset[str(my_dataset.keys()[i])]["total_stock_value"] == "NaN":
			my_dataset[str(my_dataset.keys()[i])]["stock_ratio"] = "NaN"
		else:
			my_dataset[str(my_dataset.keys()[i])]["stock_ratio"] = 1.0
	else:
		my_dataset[str(my_dataset.keys()[i])]["normalized_payments"] = my_dataset[str(my_dataset.keys()[i])]["total_payments"] - \
		my_dataset[str(my_dataset.keys()[i])]["loan_advances"] - my_dataset[str(my_dataset.keys()[i])]["expenses"] - \
			my_dataset[str(my_dataset.keys()[i])]["deferred_income"] - my_dataset[str(my_dataset.keys()[i])]["deferral_payments"]
		if my_dataset[str(my_dataset.keys()[i])]["normalized_payments"] == 0:
			if my_dataset[str(my_dataset.keys()[i])]["total_stock_value"] == "NaN":
				my_dataset[str(my_dataset.keys()[i])]["stock_ratio"] = "NaN"
			else:
				my_dataset[str(my_dataset.keys()[i])]["stock_ratio"] = 1.0
		else:
			if my_dataset[str(my_dataset.keys()[i])]["total_stock_value"] == "NaN":
				my_dataset[str(my_dataset.keys()[i])]["stock_ratio"] = 0.0
			else:
				my_dataset[str(my_dataset.keys()[i])]["stock_ratio"] = float(my_dataset[str(my_dataset.keys()[i])]["total_stock_value"]) / \
				(float(my_dataset[str(my_dataset.keys()[i])]["normalized_payments"]) + float(my_dataset[str(my_dataset.keys()[i])]["total_stock_value"]))
	
### tested an outlier
# stock_points = []
# normal_points = []
# for i in my_dataset:
# 	stock_points.append(my_dataset[i]["stock_ratio"])
# 	normal_points.append(my_dataset[i]["normalized_payments"])

# for i in range(0, len(stock_points)):
# 	if stock_points[i] == 'NaN':
# 		stock_points[i] = 0.0

# plt.scatter(stock_points, normal_points)
# plt.axis([min(stock_points), max(stock_points), min(normal_points), max(normal_points)])
# plt.show()

# print my_dataset.keys()[4]
# print my_dataset['HANNON KEVIN P']['stock_ratio']
# print my_dataset['HANNON KEVIN P']['normalized_payments']
### found a problem with the creation of normalized_payments. This is no longer an outlier


email_features = ['to_messages', 'from_poi_to_this_person', 
'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']

for i in range(len(my_dataset.keys())):
	for j in email_features:
		if my_dataset[str(my_dataset.keys()[i])][j] == 'NaN':
			my_dataset[str(my_dataset.keys()[i])][j] = 0.0
	if my_dataset[str(my_dataset.keys()[i])]['to_messages'] == 0.0:
		my_dataset[str(my_dataset.keys()[i])]['in_poi_percentage'] = 0.0
	else:
		my_dataset[str(my_dataset.keys()[i])]['in_poi_percentage'] = my_dataset[str(my_dataset.keys()[i])]['from_poi_to_this_person'] / float(my_dataset[str(my_dataset.keys()[i])]['to_messages'])
	if my_dataset[str(my_dataset.keys()[i])]['from_messages'] == 0.0:
		my_dataset[str(my_dataset.keys()[i])]['out_poi_percentage'] = 0.0
	else:
		my_dataset[str(my_dataset.keys()[i])]['out_poi_percentage'] = my_dataset[str(my_dataset.keys()[i])]['from_this_person_to_poi'] / float(my_dataset[str(my_dataset.keys()[i])]['from_messages'])

### Check new features	
# print my_dataset['KAMINSKI WINCENTY J']
# print my_dataset['MARTIN AMANDA K']
# print my_dataset['SHAPIRO RICHARD S']
# print my_dataset['KEAN STEVEN J']

### Look at outliers in new features
# for i in my_dataset:
# 	if my_dataset[i]["out_poi_percentage"] > 0.8:
# 		print "high emails to pois:", i, my_dataset[i]

# for i in my_dataset:
# 	if my_dataset[i]["in_poi_percentage"] > 0.15:
# 		print "high emails from pois:", i, my_dataset[i]


my_feature_list = ['normalized_payments', 'stock_ratio', 'in_poi_percentage', 'out_poi_percentage']

# for i in my_feature_list:
# 	features_list.append(i)

# features_list.append('normalized_payments')
features_list.append('stock_ratio')
# features_list.append('in_poi_percentage')
features_list.append('out_poi_percentage')



# print features_list
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


print "my_dataset length", len(my_dataset)
print "features_list length", len(features_list)
print "data length", len(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
from sklearn.cross_validation import train_test_split, KFold

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.31, random_state=42)

#pca.fit(features_train)

# features_train = pca.fit_transform(features_train)
# features_test = pca.fit_transform(features_test)
from sklearn.decomposition import PCA
from sklearn import svm, grid_search
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


###Cross Validation Technique (choose one)
cv = KFold(len(data), 4)
# cv = 4

###Dimensionality Reduction with PCA
pca = PCA()

svm = svm.LinearSVC(random_state=42, tol=0.1, penalty = 'l2', loss = 'squared_hinge') 
	#,   

kbest = SelectKBest(f_classif)
	#, k=14

scaler = StandardScaler(with_mean = True, with_std = False, copy = True)
	#

pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('kbest', kbest), ('svm', svm)])

# print 'svm parameters:', pipe.get_params().keys()

params = {'svm__max_iter':[18], \
'kbest__k':[3], \
'pca__n_components':[3]} 
### Removed param options: 'penalty': ('l1'), 'loss':('squared_hinge')
#'svm__max_iter':[1:20,10-100 every 5, 100, 1000, 10000], 'kbest__k':[2,3,4,6,7,8,9,10,11],
#'pca__n_components':[1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22],
#  'svm__tol':[.1, .01, .001,.0001,.00001]

clf_svm = grid_search.GridSearchCV(pipe, params, cv=5, error_score = 0, scoring = 'recall')

### Try Another Classifier
knn = KNeighborsClassifier(algorithm = 'auto', n_jobs = -1, metric='chebyshev', weights = 'distance', n_neighbors = 2)
	

knn_pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('kbest', kbest), ('knn', knn)]) 

print 'knn parameters:', knn_pipe.get_params().keys()

knn_params = {'pca__n_components':[14,15,16], \
'kbest__k':[14,15,16]}
### Removed param options: 'knn__metric': ('wminkowski', 'seuclidean', 'mahalanobis', 'chebyshev', 'minkowski', 'euclidean'),
# 'knn__n_neighbors': [1,2,3,4,5,6,7,8,9,10], 'knn__weights': ('distance', 'uniform'), 
# 'pca__n_components':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23],
# 'kbest__k':[2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17, 18, 'all'],
# 'scaler__with_mean':(True, False), 'scaler__with_std':(True, False), 'scaler__copy':(True,False), 


clf_knn = grid_search.GridSearchCV(knn_pipe, knn_params, cv=5, error_score = 0, scoring = 'recall')

### Choose a Classifier
clf = clf_svm
# clf = clf_knn

clf.fit(features_train, labels_train)

print clf.best_params_

#Top PCs with scores for SVM
for i in range(0,len(clf_svm.best_estimator_.named_steps['kbest'].get_support())):
	if clf_svm.best_estimator_.named_steps['kbest'].get_support()[i] == True:
		print i, clf_svm.best_estimator_.named_steps['kbest'].scores_[i]

#Top PCs with scores for KNN
# for i in range(0,len(clf_knn.best_estimator_.named_steps['kbest'].get_support())):
# 	if clf_knn.best_estimator_.named_steps['kbest'].get_support()[i] == True:
# 		print i, clf_knn.best_estimator_.named_steps['kbest'].scores_[i]

pred = clf.predict(features_test)

print "features_train:",len(features_train)
print "labels_train:",len(labels_train)
print "features_test:",len(features_test)
print "labels_test:",len(labels_test)

# print "PCA Components:", PCA().fit(data).components_



# Pass the best algorithm as your clf
clf = clf.best_estimator_


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!




from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix, roc_auc_score
print "Accuracy:", accuracy_score(labels_test, pred)
print clf.score(features_test, labels_test)

print classification_report(labels_test, pred)


print "precision:", precision_score(labels_test, pred)
print "recall:", recall_score(labels_test, pred)
print "F1:", f1_score(labels_test, pred)

print "MSE:", mean_squared_error(labels_test, pred)
print "RMSE:", mean_squared_error(labels_test, pred)**0.5

print "ROC_AUC:", roc_auc_score(labels_test, pred)

print "Confusion Matrix:", confusion_matrix(labels_test, pred)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
