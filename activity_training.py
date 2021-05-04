import numpy as np
import os
import csv
from util import slidingWindow
from features import extract_features
from sklearn.model_selection import KFold
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

#--------------------------------------------------
#
#      Load files and concatenate into one file
#
#--------------------------------------------------

print('Currently loading data into a singular file... ', end = '')
class_names = ['sitting', 'standing', 'laying', 'walking', 'falling']
dir_name = './data/trainingData'
output_file = open('activity.csv', 'w')
for filename in os.listdir(dir_name):
    try:
        if 'fall' in filename:
            class_index = class_names.index('falling')
        else:
            class_index = class_names.index(filename[:-4])
    except:
        continue
    f = open('data/trainingData/' + filename, 'r')
    f.readline() #remove separator
    f.readline() #remove header
    for line in f:
        line = line.strip()
        line += ';' + str(class_index) + '\n'
        output_file.write(line)
    f.close()
output_file.close()
print('Done!')

#--------------------------------------------------
#
#           Generate np array from file
#           and begin extracting features
#
#--------------------------------------------------

print('Beginning to extract features from the data... ', end = '')
data_file_name = 'activity.csv' #change later
data = np.genfromtxt(data_file_name, delimiter = ';')

window_size = 20
step_size = 20
X = []
Y = []

for i, window in slidingWindow(data, window_size, step_size):
    data_window = np.delete(window, (3,4,5), axis = 1) #take only the x,y,z data
    feature_names, x = extract_features(data_window)
    X.append(x)
    Y.append(window[window_size//2, -1])

X = np.asarray(X)
Y = np.asarray(Y)
print('Done!')

#--------------------------------------------------
#
#        Begin training different classifiers
#
#--------------------------------------------------

print('Beginning to train various different depths of decision trees...')
cv = KFold(n_splits = 10, random_state = None, shuffle = True)

print('Depth\t|\tAccuracy\t\t|\tPrecision\t\t|\tRecall')
for depth in range(1,8): #trying different depths
    acc = []
    prec = []
    rec = []
    tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = depth)
    for train, test in cv.split(X, Y):
        tree.fit(X[train], Y[train])
        prediction = tree.predict(X[test])
        acc.append(accuracy_score(Y[test], prediction))
        prec.append(precision_score(Y[test], prediction, average = 'weighted'))
        rec.append(recall_score(Y[test], prediction, average = 'weighted'))
    print(f'{depth}\t|\t{np.mean(acc):.15f}\t|\t{np.mean(prec):.15f}\t|\t{np.mean(rec):.15f}')

#--------------------------------------------------
#
#        Save all trees and export to pickle
#
#--------------------------------------------------

for i in range(1,8):
    allTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = i)
    allTree.fit(X, Y)
    export_graphviz(allTree, out_file = f'dots/tree_{i}.dot', feature_names = feature_names)
    with open(f'pickles/classifier_{i}.pickle', 'wb') as f:
        pickle.dump(allTree, f)