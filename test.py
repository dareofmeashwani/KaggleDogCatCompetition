import sys
from Model import Network
from Trainer import Trainer
import numpy as np
from ImageOps import ImageOps
imageOps = ImageOps()

batchsize = 32
imageSize = 64

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

trainer = Trainer(model)
trainer.batchSize = 32
trainer.workingDir = "SavedModelCNN"


img=imageOps.readImage(["test1.jpg","test2.jpg"])
img= imageOps.resizeImageWithAspectRatio(img,imageSize)
img=np.array(img).reshape([-1,imageSize,imageSize,3])
print(img.shape)
test={
    "x":img
}

print(trainer.predict(test))


