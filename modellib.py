import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


n_steps = 5
n_features = 1

def splitSequence(seq, n_steps):
    
    #Declare X and y as empty list
    X = []
    y = []
    for i in range(len(seq)):
        #get the last index
        lastIndex = i + n_steps
        
        #if lastIndex is greater than length of sequence then break
        if lastIndex > len(seq) - 1:
            break
            
        #Create input and output sequence
        seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]
        
        #append seq_X, seq_y in X and y list
        X.append(seq_X)
        y.append(seq_y)        
        pass    #Convert X and y into numpy array
    X = np.array(X)
    y = np.array(y)


    return X,y 
    
    pass


def getBlock(addr):
    return (addr >> 6) << 6

def isSameBlock(addr1, addr2):
    return getBlock(addr1) == getBlock(addr2)

def createTrainData(array, trace):
    with open(trace) as file:
        list = []
        for line in file:
            pos, type, value = line.split(" ")
            value = (int(value, base=16))
            list.append(value)
        maxi = max(list)
        mini = min(list)
        for i in list:
            final = (i - mini)/(maxi)
            array.append(final)
            np.array(array)
        return array