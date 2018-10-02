#!usr/bin/env python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import csv

# Load training data
data = np.load('train.npy')
X = np.matrix.transpose( data[1:] )
y = data[0]
# Load test data
test = np.load('test.npy')
X_test = np.matrix.transpose( test[1:] )

rf = RandomForestClassifier()

# Grid search to optimize forest hyperparameters
grid = {'min_samples_leaf':[3],
            'n_estimators':[100],
            'max_features':[3],
               'bootstrap':[False]}
gridsearch = GridSearchCV(estimator = rf, 
                         param_grid = grid,
                         cv = 5,
                         n_jobs = 5,
                         verbose=3)
gridsearch.fit(X, y)
#print gridsearch.best_params_

# predict test target: survival of test-case passengers
pred = gridsearch.predict(X_test)

# write predictions to submission csv
with open('my_submission.csv', 'w') as csvfile:
  writer = csv.writer( csvfile, delimiter=',')
  writer.writerow(['PassengerId','Survived'])
  for i in range(len(test[0])):
    writer.writerow([int(test[0][i]), int(pred[i])])

