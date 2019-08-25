import sys

sys.path.insert(0, '../')
from DataQueue import DataQueue
from Model import Vgg16
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
for _ in range(20):
    x, y = testGen.batchProcessor(*tuple(testGen.batchGenerator()))
    test['x'].append(np.array(x))
    test['y'].append(np.array(y))

test['x'] = np.concatenate(test['x'], axis=0)
test['y'] = np.concatenate(test['y'], axis=0)
print(test['x'].shape, test['y'].shape)

trainDq = DataQueue(trainGen, childCount=3, qsize=8)

model = Vgg16()
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
    trainer = Trainer(model,"adam")
    trainer.batchSize = 32
    trainer.steps = 1000
    trainer.restoreState = True
    trainer.maxKeep = 25
    trainer.learningRate = 0.000001
    trainer.trainPrintFreq = 1
    trainer.workingDir = "vggDogCat"
    trainer.train(trainDq, test, valPrintFreq=25)
finally:
    trainDq.stop()
    pass

