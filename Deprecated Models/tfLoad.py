import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
constantDivider = 1e7

# checkpointPage = "savedModels/checkpointPage"
# checkpoint_dir = os.path.dirname(checkpointPage)
# checkpointOffset = "savedModels/checkpointOffset"
# checkpoint_dir = os.path.dirname(checkpointOffset)


# # Create a callback that saves the model's weights
# pageCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPage,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# offsetCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointOffset,
#                                                  save_weights_only=True,
#                                                  verbose=1)

n_steps = 5
n_features = 1
offsetModel = tf.keras.models.load_model('savedModels/savedOffsetModel')
pageModel = tf.keras.models.load_model('savedModels/savedPageModel')
offsetModel.load_weights('checkpoints/checkpointOffset')
pageModel.load_weights('checkpoints/checkpointPage')



###############################PAGE#########################################


pageModel.summary()


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
pagePrediction = abs((int(pagePredict))*constantDivider)
pageActual = 140724950000000

#################################offset#######################################
offsetData = []
constantDivider2 = 1e7


offsetModel.summary()

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
offsetPrediction = abs(int(predictNextNumber))
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