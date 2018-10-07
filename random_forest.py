#!usr/bin/env python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import csv

# Load training data
data = np.load('train_features.npy')
data = data.item()
X_dict = {}
for key in data.keys():
  if key not in ['PassengerID', 'Survived']: X_dict[key] = data[key]
X = [ [X_dict[key][i] for key in X_dict.keys()] for i in range(len(X_dict['Age'])) ]
y = data['Survived']

# Load test data
data_ = np.load('test_features.npy')
data_ = data_.item()
X_dict_ = {}
for key in data_.keys():
  if key not in ['PassengerID', 'Survived']: X_dict_[key] = data_[key]
X_test = [ [X_dict_[key][i] for key in X_dict_.keys()] for i in range(len(X_dict_['Age'])) ]
y_test = data_['Survived']
y_id = data_['PassengerID']

rf = RandomForestClassifier()

# Grid search to optimize forest hyperparameters
grid = {'min_samples_leaf':[3],
            'n_estimators':[100],
               'max_depth':[12],
               'bootstrap':[False]}
gridsearch = GridSearchCV(estimator = rf, 
                         param_grid = grid,
                         cv = 3,
                         n_jobs = 5,
                         verbose=3)
gridsearch.fit(X, y)
#print gridsearch.best_params_

# predict test target: survival of test-case passengers
pred = gridsearch.predict(X_test)

# write predictions to submission csv
with open('my_submission_v2.csv', 'w') as csvfile:
  writer = csv.writer( csvfile, delimiter=',')
  writer.writerow(['PassengerId','Survived'])
  for i in range(len(y_id)):
    writer.writerow([int(y_id[i]), int(pred[i])])


