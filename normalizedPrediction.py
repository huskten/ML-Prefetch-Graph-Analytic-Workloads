import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
constantDivider = 1e7

Ncheckpoint = "Ncheckpoints/Ncheckpoint"
checkpoint_dir = os.path.dirname(Ncheckpoint)


# Create a callback that saves the model's weights
Callback = tf.keras.callbacks.ModelCheckpoint(filepath=Ncheckpoint,
                                                 save_weights_only=True,
                                                 verbose=1)


########################################################################

Data = []

def createTrainData(array):
    with open('gimpedpinatracemedium.out') as file:
        list = []
        for line in file:
            pos, type, value = line.split(" ")
            value = (int(value, base=16))
            list.append(value)
        minTrain = min(list)
        maxTrain = max(list)
        for i in list:
            final = (i - minTrain)/(maxTrain)
            array.append(final)
            np.array(array)
        return array

createTrainData(Data)


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

n_steps = 5

A, b = splitSequence(Data, n_steps = 5)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
A = A.reshape((A.shape[0], A.shape[1], n_features))

Model = tf.keras.Sequential()
Model.add(layers.LSTM(256, activation='relu', input_shape=(n_steps, n_features)))
Model.add(layers.Dense(1))

Model.layers

#Model.summary()

Model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError())

Model.fit(A, b, epochs=5, verbose=1)

PredictArr = []

Model.save('savedModels/normModel')

def createPredictData(array):
    with open('gimpedpinatracepredict.out') as File:
        list = []
        for line in File:
            predictPos, predictType, predictValue = line.split(" ")
            predictValue = (int(predictValue, base=16))
            list.append(predictValue)
        global minPredict 
        minPredict = min(list)
        global maxPredict 
        maxPredict = max(list)
        for i in list:
            finalPredict = (i - minPredict)/(maxPredict)
            array.append(finalPredict)
        return array
            
createPredictData(PredictArr)

PredictArr = np.array(PredictArr)

PredictArr = PredictArr.reshape((1, n_steps, n_features))

PredictArr


Predict = (Model.predict(PredictArr, verbose=1))

############################################################
# ATTEMPTS AT LOOPING PREDICTIONS
# def predictNext(array, amount):
#     for i in range(amount):
#         createPredictData(array.append(float(Predict)))
#         Model.predict(PredictArr, verbose=1)
#     return array

# predictList = []

# print(predictNext(predictList, 2))
############################################################

print("=====================================")


print(float(Predict))
formattedFinal = int(((float(Predict))*(maxPredict - minPredict) + minPredict))
print(formattedFinal, 140724957118656)
print(formattedFinal - 140724957118656)
print(formattedFinal == 140724957118656)

