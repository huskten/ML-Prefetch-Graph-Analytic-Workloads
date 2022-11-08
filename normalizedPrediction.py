import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
constantDivider = 1e7
np.set_printoptions(floatmode = 'unique')

# Ncheckpoint = "Ncheckpoints/Ncheckpoint"
# checkpoint_dir = os.path.dirname(Ncheckpoint)


# # Create a callback that saves the model's weights
# Callback = tf.keras.callbacks.ModelCheckpoint(filepath=Ncheckpoint,
#                                                  save_weights_only=True,
#                                                  verbose=1)

#performance
#model accuracy
#prefetch accuracy
#area takes
#speed up
#Sniper implementation
#how long is too long
#prediodic vs smart

########################################################################

Model = tf.keras.models.load_model('savedModels/normModel')


PredictData = []
n_steps = 5
n_features = 1

def splitSequence(seq, n_steps):
    
    #Declare X and y as empty list
    X = []
    y = []
    Data
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

def createPredictData(array):
    temp = []
    with open('gimpedpinatracepredict.out') as File:
        for line in File:
            predictPos, predictType, predictValue = line.split(" ")
            predictValue = (int(predictValue, base=16))
            temp.append(predictValue)
        global minPredict 
        minPredict = temp[50]
        global maxPredict 
        maxPredict = max(temp)
        for i in temp:
            final = (i - minPredict)/(maxPredict - minPredict)
            array.append(final)
        return array

def iteratePredictData(array):
    # for i in PredictData:
    #         finalPredict = (i - minPredict)/(maxPredict)
    #         np.append(array, finalPredict)
    #         if len(array) > 5:
    #             array.pop(0)
    #             break
    for i in range(len(array)):
            # t = []
            # np.array(t)
            t = array[i:i+5]
            t = np.array(t, dtype = np.longdouble)
            t = t.reshape((1, n_steps, n_features))
            prediction = float(Model.predict(t, verbose = 0))
            print(prediction)
            print((((prediction)*(maxPredict - minPredict)) + minPredict))
    return array

createPredictData(PredictData)

print(iteratePredictData(PredictData))