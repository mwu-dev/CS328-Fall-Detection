import numpy as np
from matplotlib import pyplot as plt

def removeHeaders(filename, out_file):
    f_out = open(out_file, 'w')
    f = open(filename, 'r')
    f.readline() #skip seperator
    f.readline() #skip header
    for line in f:
        f_out.write(line)
    f.close()
    f_out.close()

def graph(filename):
    removeHeaders(filename, out_file = 'graphdata.csv')

    data = np.genfromtxt('graphdata.csv', delimiter = ';')
    coords = data[:,:3]
    magn = np.array([np.linalg.norm(val[:3]) for val in coords])
    timestamps = data[:,3]
    # coords = data[:, 1:4]
    # timestamps = data[:,0]

    plt.figure(figsize = (10,5))
    plt.plot(timestamps, magn)
    plt.show()
