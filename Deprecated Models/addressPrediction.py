import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
constantDivider = 1e7

checkpointPage = "checkpoints/checkpointPage"
checkpoint_dir = os.path.dirname(checkpointPage)
checkpointOffset = "checkpoints/checkpointOffset"
checkpoint_dir = os.path.dirname(checkpointOffset)


# Create a callback that saves the model's weights
pageCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPage,
                                                 save_weights_only=True,
                                                 verbose=1)

offsetCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointOffset,
                                                 save_weights_only=True,
                                                 verbose=1)

###############################PAGE#########################################

pageData = []

with open('gimpedpinatracemedium.out') as file:
    for line in file:
        pos, type, value = line.split(" ")
        pageData.append((int(value, base=16))/constantDivider)
        
def splitSequence(seq, n_steps):
    
    #Declare X and y as empty list
    X = []
    y = []
    pageData
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

A, b = splitSequence(pageData, n_steps = 5)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
A = A.reshape((A.shape[0], A.shape[1], n_features))

pageModel = tf.keras.Sequential()
pageModel.add(layers.LSTM(512, activation='relu', input_shape=(n_steps, n_features)))
pageModel.add(layers.Dense(1))

pageModel.layers

#pageModel.summary()

pageModel.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError())

pageModel.fit(A, b, epochs=150, verbose=1, callbacks=[pageCallback])

pagePredictArr = []

with open('gimpedpinatracepredict.out') as pageFile:
    for i in range(n_steps):
        for line in pageFile:
            predictPos, predictType, predictValue = line.split(" ")
            pagePredictArr.append((int(predictValue, base=16))/constantDivider)

pagePredictArr = np.array(pagePredictArr)

pagePredictArr = pagePredictArr.reshape((1, n_steps, n_features))

pagePredictArr

pageModel.save('savedModels/savedPageModel')


pagePredict = (pageModel.predict(pagePredictArr, verbose=1))
pagePrediction = (int(pagePredict))*constantDivider
pageActual = 140724950000000

#################################offset#######################################
offsetData = []
constantDivider2 = 1e7


with open('gimpedpinatracemedium.out') as offsettrainfile:
    for line in offsettrainfile:
        pos, type, value = line.split(" ")
        offsetData.append((int(value, base=16))%constantDivider2)


n_steps = 5

X, y = splitSequence(offsetData, n_steps = 5)



# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

offsetModel = tf.keras.Sequential()
offsetModel.add(layers.LSTM(256, activation='relu', input_shape=(n_steps, n_features)))
offsetModel.add(layers.Dense(1))

offsetModel.layers

#offsetModel.summary()

offsetModel.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError())

offsetModel.fit(X, y, epochs=200, verbose=1, callbacks=[offsetCallback])

offsetPredictArr = []

with open('gimpedpinatracepredict.out') as offsetFile:
    for i in range(n_steps):
        for line in offsetFile:
            predictPos, predictType, predictValue = line.split(" ")
            offsetPredictArr.append(((int(predictValue, base=16))%constantDivider2))

offsetPredictArr = np.array(offsetPredictArr)

offsetPredictArr = offsetPredictArr.reshape((1, n_steps, n_features))

offsetPredictArr


offsetModel.save('savedModels/savedOffsetModel')

predictNextNumber = (offsetModel.predict(offsetPredictArr, verbose=1))
offsetPrediction = (int(predictNextNumber))
offsetActual = (hex(140724957118656))
print("================Page=====================")
print(pagePredict)
print(pagePrediction)
print(int(pagePredict))
print("================offset=====================")
print(predictNextNumber)
print(int(predictNextNumber))
print("=====================================")
print("pagePrediction:", pagePrediction)
print("pageActual:", 140724950000000)
print("offsetPrediction:", offsetPrediction)
print("actualoffset:", 140724957118656 - 140724950000000)
finalPrediction = (offsetPrediction + pagePrediction)
print("finalPrediction:", finalPrediction, "Length:", len(str(finalPrediction))-2)
print("actualFinal:", 140724957118656, "Length:", len(str(140724957118656)))
print("Difference:", 140724957118656 - finalPrediction)
print("True?:", finalPrediction == 140724957118656)

offsetModel.summary()
pageModel.summary()

