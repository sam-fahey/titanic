#!usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import csv, sys

# header : contains names of features in data
# X : matrix, each row is an array of m instances of a single feature
# if 'train': formats features and plots correlation matrix;
# if 'test' : formats features; exits;
filename = 'test.csv'
with open(filename, 'rb') as csvfile:
  data = csv.reader(csvfile, delimiter=',')
  X = [row for row in data]
  header = X[0]
  X = np.matrix.transpose(np.array(X[1:]))
# if converting test-file into numpy array, need a placeholder for 'survived'
if filename[:4] == 'test': 
  X = np.vstack( [X[0], X] )
  header = np.r_[['header'], header] 

# removing bad/irrelevant features (some may be useful later)
#     : "Name", "Ticket", "Cabin"
# survived : boolean 1s and 0s 
train = np.array( [ X[i] for i in [2,4,5,6,7,9,11] ] )
train_header = [ header[i] for i in [2,4,5,6,7,9,11] ]
survived = np.array( [ int(x1) for x1 in X[1] ] )

# Pclass -- all values integers
Pclass = np.array( [ int(i) for i in train[0] ] )

# Sex -- all values 'male' or 'female'
Sex = np.array( [ 1 if i=='male' else 0 for i in train[1] ] )

# Age -- some values empty, rest floats
# for now, assign mean of present values to empties
has_age = train[2] != ''
temp_age = [float(age) for age in train[2][has_age]]
mean_age = np.mean(temp_age)
Age = np.array( [ float(train[2][i]) if has_age[i] else float(mean_age) for i in range(len(train[2])) ] )
has_age = np.array( [ 1 if i else 0 for i in has_age ] )

# SibSp -- integer number of siblings and spouses aboard
SibSp = np.array( [ int(i) for i in train[3] ] )

# Parch -- integer number of parents and children aboard
Parch = np.array( [ int(i) for i in train[4] ] )

# Fare -- float cost of ticket
# test sample has one bad value, let's call it a free ticket
for i in range(len(train[5])): 
  if train[5][i] == '': train[5][i]='0'
Fare = np.array( [ float(i) for i in train[5] ] )

# Embarked -- convert letters to int
# no correlation of missing value with survival
Embarked = np.zeros(len(train[6]))
for i in range(len(train[6])):
  if     train[6][i] == 'C': Embarked[i] = 1
  elif   train[6][i] == 'Q': Embarked[i] = 2
  elif   train[6][i] == 'S': Embarked[i] = 3
  else                     : Embarked[i] = 0

# build new features
is_child = np.array( [ 1 if Age[i] < 16 else 0 for i in range(len(Age)) ] )
allfam = np.array( [ SibSp[i]+Parch[i] for i in range(len(SibSp)) ] )

# assemble all features, total includes survival at row=0
# all_<name> includes new features like "has_age" and "is_child"
features = np.array([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked])
all_features = np.array([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, has_age, is_child, allfam])
total = np.vstack( [survived, features] )
all_total = np.vstack( [survived, all_features] )
names = np.r_[['survived'], train_header]
all_names = np.r_[['survived'], train_header, ['has_age', 'is_child', 'all_fam']]
# save array for classification
np.save('%s.npy'%filename[:-4], all_total)

if filename[:-4] != 'train': print "Exit without plotting."; sys.exit()
# plot correlation matrix
corr = np.corrcoef(all_total)
cmap = plt.cm.get_cmap('RdBu_r')
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cax = ax.matshow( corr, vmin=-1, vmax=1, cmap=cmap)
fig.colorbar(cax)
ax.xaxis.tick_top()
ax.set_xticks(range(len(all_names)))
ax.set_yticks(range(len(all_names)))
ax.set_xticklabels(all_names, rotation=90)
ax.set_yticklabels(all_names)
for i in range(len(corr)):
  for j in range(len(corr[i])):
    ax.text(j, i, '%.2f'%corr[i, j], ha='center', va='center', fontsize=10)
ax.axvline(0.5, color='k'); ax.axhline(0.5, color='k')
ax.axvline(7.5, color='k', linestyle='--'); ax.axhline(7.5, color='k', linestyle='--')
plt.tight_layout()
plt.savefig('/home/sfahey/public_html/titanic_corr.pdf')
plt.savefig('/home/sfahey/public_html/titanic_corr.png')

