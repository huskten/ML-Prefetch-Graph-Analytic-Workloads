import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
constantDivider = 1e7
np.set_printoptions(floatmode = 'unique')

from modellib import splitSequence, getBlock, isSameBlock


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
#iterate through predict input trace and compare block results
#Ahmed will provide getblock function 



#drug discovery
#weather prediction
#games to load faster
#zoom in -> make either processor faster, memory, etc -> make processor faster by making prefetecher more efficient -> goal of hardware prefetcher is to bring data to processor in advance removes the delay -> prefetechers already exist sometimes -> we want a more effective prefetcher to fetch a broader range of data

#performance matters -> specific sisues of prefetechers -> why prefetcher better
#removal of the human mind automate the design process
#use hardawre to do machine learning

#set of addresses, check if next address exists
#compare adress to the set of predictions, dont compare a prediction to the set of the actual addresses

########################################################################

Model = tf.keras.models.load_model('savedModels/normModel')

PredictData = []
n_steps = 5
n_features = 1
blocks_predicted = set()

def createPredictData(array):
    temp = []
    with open('gimpedpinatracepredict.out') as File:
        for line in File:
            predictPos, predictType, predictValue = line.split(" ")
            predictValue = (int(predictValue, base=16))
            temp.append(predictValue)
        # print(temp)
        print(len(temp))
        global minPredict 
        minPredict = temp[20]
        global maxPredict 
        maxPredict = max(temp)
        for i in temp:
            final = (i - minPredict)/(maxPredict - minPredict)
            array.append(final)
        return array

def iteratePredictData(array, blockset):
    # temp = []
    hit = 0
    miss = 0
    for i in range(len(array)):
            t = array[i:i+5]
            if len(t) < 5: #temporary: change once we start predicting multiple sets of addresses and can change the amount of indices we take in
                break
            if i+6 > len(array)-1:
                break
            t = np.array(t, dtype = np.longdouble)
            t = t.reshape((1, n_steps, n_features))
            prediction = float(Model.predict(t, verbose = 0))
            fprediction = int((((prediction)*(maxPredict - minPredict)) + minPredict))
            t2 = int((((array[i+6])*(maxPredict - minPredict)) + minPredict))
            print(fprediction)
            blocks_predicted.add(getBlock(fprediction))
            if getBlock(t2) in blockset:
                hit += 1
                print("hit")
            else:
                miss += 1
                print(("miss"))
            # temp.append(int(fprediction))
            # print(temp)
    print("Hit Count: %d" % hit)
    print("Miss Count: %d" % miss)
    hitRate = (int(hit) / (int(miss)+int(hit)))*100
    print("Hit Rate:", hitRate)
    return array, blockset

createPredictData(PredictData)

iteratePredictData(PredictData, blocks_predicted)

print(blocks_predicted)

# for i in blocks_predicted:
#     return isSameBlock(i, )

# def testBlock(set):
#     with open('gimpedpinatracepredict.out') as File:
#             for line in File:
#                 predictPos, predictType, predictValue = line.split(" ")
#                 predictValue = (int(predictValue, base=16))
#                 isSameBlock(predictValue, )
