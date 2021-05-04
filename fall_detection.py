import numpy as np
import pickle
import sys
import matplotlib
import matplotlib.pyplot as plt
from cutdata import cut_data
from features import extract_features
from util import slidingWindow
from graph_util import removeHeaders

#--------------------------------------------------
#
#            Load data from test file
#
#--------------------------------------------------

print('Currently loading given data... ', end = '')
data_file_name = './data/testData/seriousFall.csv'
out_file_name = 'test.csv'
removeHeaders(data_file_name, out_file = out_file_name)

data = np.genfromtxt(out_file_name, delimiter = ';')
data = cut_data(data, threshold = 2000)
print('Done!')

#--------------------------------------------------
#
#            Load the best decision tree
#
#--------------------------------------------------

best_tree_num = 5 #CHANGE IF NEEDED
decision_tree_classifier = pickle.load(open(f'pickles/classifier_{best_tree_num}.pickle', 'rb'))

if decision_tree_classifier is None:
    print('No classifier detected.')
    sys.exit()

#--------------------------------------------------
#
#         Setup the tree's prediction list
#
#--------------------------------------------------

window_size = 20
step_size = 20

class_names = ['sitting', 'standing', 'laying', 'walking', 'falling']
color_names = ['b', 'g', 'r', 'm', 'k']

predictionList = []
for i, window in slidingWindow(data, window_size, step_size):
    data_window = np.delete(window, (3,4), axis = 1) #take only the x,y,z data
    feature_names, x = extract_features(data_window)
    y = int(decision_tree_classifier.predict([x])[0])
    for j in range(i, i + window_size):
        predictionList.append(y)

#--------------------------------------------------
#
#       Setup for plotting the data with labels
#
#--------------------------------------------------

timestamps = (data[:,3])/1000
data = np.delete(data, (3,4), axis = 1) #delete timestamps off data
magn = np.array([np.linalg.norm(val[:3]) for val in data])
magn /= np.mean(magn)

plt.figure(figsize=(10,5))
curr = int(predictionList[0]) #current activity
curr_index = 0 #current index

#stuff for legend
plots = [] #list of unique event plots
labels = [] #list of labels
usedColors = [] #list of used colors

numFalls = 0
seriousFall = False
fallIndex, standIndex, walkIndex = class_names.index('falling'), class_names.index('standing'), class_names.index('walking')
for i, event in enumerate(predictionList):
    if event == fallIndex and curr != fallIndex:
        print(f'Fall detected at time: {timestamps[i]}')
        numFalls += 1
        seriousFall = True
    elif curr == fallIndex and (event ==  standIndex or event == walkIndex):
        seriousFall = False
    if event != curr:
        line = plt.plot(timestamps[curr_index:i-1], magn[curr_index:i-1], color_names[curr])
        plt.plot(timestamps[i-2:i+1], magn[i-2:i+1], color_names[curr])
        if line[0].get_color() not in usedColors:
            usedColors.append(line[0].get_color())
            labels.append(class_names[curr])
            plots.append(line[0])
        curr = event
        curr_index = i
line = plt.plot(timestamps[curr_index:i-1], magn[curr_index:i-1], color_names[curr])
plt.plot(timestamps[i-2:i+1], magn[i-2:i+1], color_names[curr])
if line[0].get_color() not in usedColors:
    usedColors.append(line[0].get_color())
    labels.append(class_names[curr])
    plots.append(line[0])

if curr == fallIndex: #if data ends prematurely, assume fall was serious, or if previous condition holds
    seriousFall = True

plt.title("Data with Color Coded Activity")
plt.legend(plots, labels, loc = 'upper right')
plt.xlabel('Time')
plt.grid()
plt.show()

print(f'Number of falls detected: {numFalls}')
print(f'Serious fall detected: {seriousFall}')