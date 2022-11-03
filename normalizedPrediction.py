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

Model.fit(A, b, epochs=1, verbose=1)

PredictSect = []

Model.save('savedModels/normModel')


PredictSect = []
Set = []

def createPredictData(array):
    with open('gimpedpinatracepredict.out') as File:
        for line in File:
            predictPos, predictType, predictValue = line.split(" ")
            predictValue = (int(predictValue, base=16))
            Set.append(predictValue)
        global minPredict 
        minPredict = min(Set)
        global maxPredict 
        maxPredict = max(Set)
        for i in Set:
            finalPredict = (i - minPredict)/(maxPredict)
            if len(array) > 5:
                array.pop(0)
                break
            if finalPredict > 0:
                array.append(finalPredict)
        return array

def iteratePredictData(array):
    for i in Set:
            finalPredict = (i - minPredict)/(maxPredict)
            np.append(array, finalPredict)
            if len(array) > 5:
                array.pop(0)
                Set.pop(0)
                break
    return array
            
def nextPredict(array):
    iteratePredictData(array)

    array = np.array(array)

    array = array.reshape((1, n_steps, n_features))

    array

    Predict = (Model.predict(array, verbose=1))
    print(Predict)
    print(int(((float(Predict))*(maxPredict) + minPredict)))
    return

            
createPredictData(PredictSect)

PredictSect = np.array(PredictSect)

PredictSect = PredictSect.reshape((1, n_steps, n_features))

PredictSect

Predict = (Model.predict(PredictSect, verbose=1))
print(Predict)
print(int(((float(Predict))*(maxPredict) + minPredict)))


nextPredict(PredictSect)
nextPredict(PredictSect)
nextPredict(PredictSect)
nextPredict(PredictSect)
nextPredict(PredictSect)



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
formattedFinal = int(((float(Predict))*(maxPredict) + minPredict))
print(formattedFinal, 140724957118656)
print(formattedFinal - 140724957118656)
print(formattedFinal == 140724957118656)

