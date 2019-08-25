import sys

sys.path.insert(0, '../')
from DataQueue import DataQueue
from Model import Network
from Trainer import Trainer
import numpy as np
from Datasets import ClassificationImageGenerator
from ImageOps import ImageOps
imageOps = ImageOps()

batchsize = 32
imageSize = 64
trainGen = ClassificationImageGenerator('/media/batman/ent/datasets/kagglecatsanddogs_3367a/images', batchsize,
                                        imageSize=64)
testGen = ClassificationImageGenerator('/media/batman/ent/datasets/kagglecatsanddogs_3367a/PetImages', batchsize,
                                       imageSize=64, batchType="sequential")
testGen.init()
test = {'x': [], 'y': []}
for _ in range(70):
    x, y = testGen.batchProcessor(*tuple(testGen.batchGenerator()))
    test['x'].append(np.array(x))
    test['y'].append(np.array(y))

test['x'] = np.concatenate(test['x'], axis=0)
test['y'] = np.concatenate(test['y'], axis=0)
print(test['x'].shape, test['y'].shape)

trainDq = DataQueue(trainGen, childCount=3, qsize=8)

model = Network()
model.noOfClasses = 2
model.hiddenUnits = 2048
model.imageShape = [imageSize, imageSize, 3]
model.addMetrics('Accuracy')
model.convActivation = "relu"
model.fcActivation = "leakyrelu"
model.regularizationCoefficient = 0.005
model.regularizationType = "l2"
model.build()

trainDq.start()
try:
    trainer = Trainer(model)
    trainer.batchSize = 32
    trainer.steps = 5000
    trainer.restoreState = True
    trainer.maxKeep = 100
    trainer.learningRate = 0.00001
    trainer.trainPrintFreq = 1
    trainer.workingDir = "SavedModelCNN"
    trainer.train(trainDq, test, valPrintFreq=2)
finally:
    trainDq.stop()
    pass

