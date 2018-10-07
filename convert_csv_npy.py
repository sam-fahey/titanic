#!usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import csv, sys, pickle

# header : contains names of features in data
# X : matrix, each row is an array of m instances of a single feature
# if 'train': formats features and plots correlation matrix;
# if 'test' : formats features; exits;
#filename = 'train.csv'
filename = 'test.csv'
with open(filename, 'rb') as csvfile:
  data = csv.reader(csvfile, delimiter=',')
  X = [row for row in data]
  header = X[0]
  X = np.matrix.transpose(np.array(X[1:]))
if filename[:4]=='test': 
  X = np.vstack( [X[0],X] )
  header = np.r_[['filler_column'], header]
# PassengerID -- int
PassengerID = np.array( [ int(x0) for x0 in X[0] ] )

# Survived -- boolean 1s and 0s 
Survived = np.array( [ int(x1) for x1 in X[1] ] )

# Pclass -- all values integers
Pclass = np.array( [ int(x2) for x2 in X[2] ] )

# Name -- string name
Name = np.array( X[3] )

# Sex -- all values 'male' or 'female'
Sex = np.array( [ 1 if x4=='male' else 0 for x4 in X[4] ] )

# Age -- some values empty, rest floats
# for now, assign mean of present values to empties
has_age = X[5] != ''
temp_age = [float(age) for age in X[5][has_age]]
mean_age = np.mean(temp_age)
Age = np.array( [ float(X[5][i]) if has_age[i] else float(mean_age) for i in range(len(X[5])) ] )

# SibSp -- integer number of siblings and spouses aboard
SibSp = np.array( [ int(x6) for x6 in X[6] ] )

# Parch -- integer number of parents and children aboard
Parch = np.array( [ int(x7) for x7 in X[7] ] )

# Ticket -- string ticket ID
Ticket = np.array( X[8] )

# Fare -- float cost of ticket
# test sample has one bad value, let's call it a free ticket
for i in range(len(X[9])): 
  if X[9][i] == '': X[9][i]='0'
Fare = np.array( [ float(x9) for x9 in X[9] ] )

# Cabin -- string cabin
Cabin = np.array( X[10] )

# Embarked -- convert letters to int
# no correlation of missing value with survival
Embarked = np.zeros(len(X[11]))
for i in range(len(X[11])):
  if     X[11][i] == 'C': Embarked[i] = 1
  elif   X[11][i] == 'Q': Embarked[i] = 2
  elif   X[11][i] == 'S': Embarked[i] = 3
  else                  : Embarked[i] = 0

all_features = np.array([
  PassengerID, Survived, Pclass, 
  Name, Sex, Age, SibSp, Parch, 
  Ticket, Fare, Cabin, Embarked, has_age])
feature_dict = {
  'PassengerID':PassengerID,
  'Survived':Survived,
  'Pclass':Pclass,
  'Name':Name,
  'Sex':Sex,
  'Age':Age,
  'SibSp':SibSp,
  'Parch':Parch,
  'Ticket':Ticket,
  'Fare':Fare,
  'Cabin':Cabin,
  'Embarked':Embarked,
  }

# save array for classification
np.save('%s.npy'%filename[:-4], feature_dict)

has_cabin = np.array( [1 if cabin != '' else 0 for cabin in Cabin ] )
has_age = np.array( [ 1 if i else 0 for i in has_age ] )
is_mrs = np.array( [ 1 if 'Mrs.' in name else 0 for name in Name ] )
is_miss = np.array( [ 1 if 'Miss.' in name else 0 for name in Name ] )
is_mr = np.array( [ 1 if 'Mr.' in name else 0 for name in Name ] )
len_name = np.array( [ len(name) for name in Name ] )

feature_dict_final = {
  'PassengerID':PassengerID,
  'Survived':Survived,
  'Pclass':Pclass,
  'is_mrs':is_mrs,
  'is_miss':is_miss,
  'is_mr':is_mr,
  'len_name':len_name,
  'Sex':Sex,
  'Age':Age,
  'has_age':has_age,
  'SibSp':SibSp,
  'Parch':Parch,
  'Fare':Fare,
  'has_cabin':has_cabin,
  'Embarked':Embarked,
  }

# save array for classification
np.save('%s_features.npy'%filename[:-4], feature_dict_final)



