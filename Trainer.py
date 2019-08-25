import tensorflow as tf
import ops
import pickle
import sys
import os
import numpy as np
import  random

try:
    import DataQueue
except:
    DataQueue = None
    print("Not able to Load DataQueue")


def printInSameLine(value):
    sys.stdout.write('\r')
    sys.stdout.flush()
    sys.stdout.write(value)
    sys.stdout.flush()


class Trainer:
    restoreState = False
    workingDir = None
    maxKeep = 20
    trainPrintFreq = 1
    batchSize = 64
    steps = 10
    learningRate = 0.001
    optimizer = "adam"
    trainRecord = []
    valRecord = []
    __valSessionOutput =None
    __trainSessionOutput =None
    __testSessionOutput =None
    __optimizer = None
    __trainCounter = 0
    __trainData = None
    __valCounter = 0
    __valData = None
    __testCounter = 0
    __testData = None
    __trainWriter = None
    __valWriter = None
    __isValRunning=False

    def __init__(self, model,optimizer="adam"):
        self.__model = model
        self.__session = tf.InteractiveSession()
        self.__merged = tf.summary.merge_all()
        self.__build(optimizer)

    def __build(self,optimizer):
        if optimizer.lower() == "adam":
            self.__optimizer = tf.train.AdamOptimizer(self.__model.inputs['learningRate']).minimize(
                self.__model.outputs['loss'])
        elif optimizer.lower() == "momentumupdate":
            self.__optimizer = tf.train.MomentumOptimizer(self.__model.inputs['learningRate'],.99).minimize(
                self.__model.outputs['loss'])

        self.__valSessionOutput= list(self.__model.outputs.values()) + [self.__merged]
        self.__trainSessionOutput= list(self.__model.outputs.values()) + [self.__merged, self.__optimizer]
        self.__testSessionOutput = [self.__model.outputs['prediction']]

        self.__metricIndex = []
        self.__trainDisplaystring = "Training::epoch:{0:6}"
        self.__valDisplaystring = "Val::"
        for i, v in enumerate(self.__model.metrics):
            self.__metricIndex.append(list(self.__model.outputs.keys()).index(v))
            self.__trainDisplaystring += '\t' + v + ':{' + str(i + 1) + ':6f}'
            self.__valDisplaystring += '\t' + v + ':{' + str(i) + ':6f}'
        self.__valDisplaystring += '\n'


    def __runDataInSession(self, feedDict, type=None):
        if type == "train":
            return tuple(self.__session.run(self.__trainSessionOutput, feed_dict=feedDict))
        elif type == "val":
            return tuple(self.__session.run(self.__valSessionOutput, feed_dict=feedDict))
        elif type == "test":
            return self.__session.run(self.__testSessionOutput, feed_dict=feedDict)

    def __getTrainBatch(self):
        if DataQueue is not None and isinstance(self.__trainData, DataQueue.DataQueue):
            return self.__trainData.getBatch()
        else:
            start = self.__trainCounter
            end = self.__trainCounter + self.batchSize
            self.__trainCounter = end
            if end > len(self.__trainData['x']):
                end = len(self.__trainData['x'])
                self.__trainCounter = 0
                ind_list = [i for i in range(len(self.__trainData['x']))]
                random.shuffle(ind_list)
                self.__trainData['x'] = self.__trainData['x'][ind_list]
                self.__trainData['y'] = self.__trainData['y'][ind_list]
            return self.__trainData['x'][start: end], self.__trainData['y'][start: end]

    def __getValBatch(self):
        if DataQueue is not None and isinstance(self.__valData, DataQueue.DataQueue):
            try:
                if self.__valData.getCurrentBatchQueueSize() > 0 or self.__valData.getCurrentJobQueueSize() > 0:
                    return self.__valData.getBatch()
                else:
                    return None
            except:
                return None, None
        else:
            start = self.__valCounter
            end = self.__valCounter + self.batchSize
            self.__valCounter = end
            if end > len(self.__valData['x']):
                end = len(self.__valData['x'])
            if start >= len(self.__valData['x']):
                return None, None
            return self.__valData['x'][start: end], self.__valData['y'][start: end]

    def __getTestBatch(self):
        if DataQueue is not None and isinstance(self.__testData, DataQueue.DataQueue):
            try:
                if self.__testData.getCurrentBatchQueueSize() > 0 or self.__testData.getCurrentJobQueueSize() > 0:
                    return self.__testData.getBatch()
                else:
                    return None
            except:
                return None
        else:
            start = self.__testCounter
            end = self.__testCounter + self.batchSize
            self.__testCounter = end
            if end > len(self.__testData['x']):
                end = len(self.__testData['x'])
            if start >= len(self.__testData['x']):
                return None
            return self.__testData['x'][start: end]

    def __checkRestore(self):
        saver = tf.train.Saver()
        print(self.workingDir + '/model/')
        try:
            saver.restore(self.__session, tf.train.latest_checkpoint(self.workingDir + '/model/'))
            return True
        except:
            return False

    def train(self, trainData, valData=None, valPrintFreq=5):
        self.__isTrainRunning = True
        assert DataQueue is not None and isinstance(trainData,DataQueue.DataQueue) or 'x' in trainData and len(trainData['x']) > 0 and 'y' in trainData and len(trainData['x']) > 0
        self.__trainData = trainData
        self.__trainCounter = 0

        init = tf.global_variables_initializer()
        self.__session.run(init)

        epoch_offset = 0
        saver = tf.train.Saver(max_to_keep=self.maxKeep)
        if self.restoreState == True and self.workingDir is not None:
            name = ops.lookForLastCheckpoint(self.workingDir + "/model/")
            if os.path.exists(os.path.join(self.workingDir, "trainResult.pickle")):
                fr = open(os.path.join(self.workingDir, "trainResult.pickle"), 'rb')
                self.trainRecord = pickle.load(fr)
                fr.close()
            if os.path.exists(os.path.join(self.workingDir, "valResult.pickle")):
                fr = open(os.path.join(self.workingDir, "valResult.pickle"), 'rb')
                self.valRecord = pickle.load(fr)
                fr.close()
            if name is not None:
                saver.restore(self.__session, self.workingDir + "/model/" + name)
                print('Model Succesfully Loaded : ', name)
                epoch_offset = int(name[6:])
        else:
            if os.path.exists(os.path.join(self.workingDir, "trainResult.pickle")):
                os.remove(os.path.join(self.workingDir, "trainResult.pickle"))
            if os.path.exists(os.path.join(self.workingDir, "valResult.pickle")):
                os.remove(os.path.join(self.workingDir, "valResult.pickle"))

        if self.workingDir is not None:
            self.__trainWriter = tf.summary.FileWriter(self.workingDir + '/train', self.__session.graph)
            self.__valWriter = tf.summary.FileWriter(self.workingDir + '/val')

        for epoch in range(epoch_offset + 1, epoch_offset + self.steps + 1):
            batchX, batchY = self.__getTrainBatch()
            feedDict = {self.__model.inputs['x']: batchX, self.__model.inputs['y']: batchY,
                        self.__model.inputs['learningRate']: self.learningRate,
                        self.__model.inputs['isTraining']: True}
            sessionResult = self.__runDataInSession(feedDict, "train")
            result = [epoch]
            for i in self.__metricIndex:
                result.append(sessionResult[i])

            if (epoch+epoch_offset) % self.trainPrintFreq == 0:
                printInSameLine(self.__trainDisplaystring.format(*(tuple(result))))
                print('')
                if self.workingDir is not None:
                    self.__trainWriter.add_summary(sessionResult[-2], epoch)
            else:
                printInSameLine(self.__trainDisplaystring.format(*(tuple(result))))
            self.trainRecord.append(result)

            if self.workingDir is not None:
                save_path = saver.save(self.__session, self.workingDir + "/model/" + 'model', global_step=epoch)

            if valData is not None and epoch % valPrintFreq == 0:
                self.__isValRunning= True
                pred , result=self.predictAndScore(valData)
                self.__isValRunning = False
                result = [ result[key] for key in result.keys()]
                self.valRecord.append(result)
                if len(result) == 1:
                    printInSameLine(self.__valDisplaystring.format(result[0]))
                else:
                    printInSameLine(self.__valDisplaystring.format(*tuple(result)))

        fw = open(os.path.join(self.workingDir, "trainResult.pickle"), 'wb')
        pickle.dump(self.trainRecord, fw)
        fw.close()
        if len(self.valRecord) != 0:
            fw = open(os.path.join(self.workingDir, "valResult.pickle"), 'wb')
            pickle.dump(self.valRecord, fw)
            fw.close()
        print('\nTraining Succesfully Complete')
        self.__isTrainRunning = False

    def predict(self, testData):
        assert testData and testData['x'].shape[0] > 0
        saver = tf.train.Saver()
        print('Retored model', ops.lookForLastCheckpoint(self.workingDir + "/model/"))
        saver.restore(self.__session, tf.train.latest_checkpoint(self.workingDir+'/model/'))
        self.__testData = testData
        self.__testCounter = 0
        prediction = None
        while True:
            batchX = self.__getTestBatch()
            if batchX is None:
                break
            feedDict = {self.__model.inputs['x']: batchX,self.__model.inputs['isTraining']: False}

            sessionResult = np.array(self.__runDataInSession(feedDict, "test"))
            if sessionResult.shape[0] == 1:
                sessionResult = sessionResult.reshape((-1))
                if prediction is None:
                    prediction = sessionResult
                else:
                    prediction = np.hstack((prediction,sessionResult))
            else:
                if prediction is None:
                    prediction = sessionResult
                else:
                    prediction = np.concatenate((prediction, sessionResult), axis=0)
        return prediction

    def predictAndScore(self,testData):

        assert 'x' in testData and testData['x'].shape[0] > 0 and 'y' in testData and testData['y'].shape[0] > 0

        if self.__isValRunning == False:
            saver = tf.train.Saver()
            saver.restore(self.__session, tf.train.latest_checkpoint(self.workingDir+'/model/'))
            print ('Retored model',ops.lookForLastCheckpoint(self.workingDir + "/model/"))

        self.__valData = testData
        self.__valCounter = 0
        prediction = None
        result = [0] * len(self.__model.metrics)
        predictionIndex = list(self.__model.outputs.keys()).index('prediction')
        batchLen= []
        while True:
            batchX,batchY = self.__getValBatch()
            if batchX is None or batchY is None:
                break
            batchLen.append(len(batchX)*1.0)
            feedDict = {self.__model.inputs['x']: batchX,self.__model.inputs['y']: batchY,self.__model.inputs['isTraining']: False}
            sessionResult = np.array(self.__runDataInSession(feedDict, "val"))

            j = 0
            for i in self.__metricIndex:
                result[j] += sessionResult[i]*batchLen[-1]
                j += 1

            if sessionResult[predictionIndex].shape[0] == 1:
                pred = sessionResult[predictionIndex].reshape((-1))
                if prediction is None:
                    prediction = pred
                else:
                    prediction = np.hstack((prediction,pred))
            else:
                if prediction is None:
                    prediction = sessionResult[predictionIndex]
                else:
                    prediction = np.concatenate((prediction,sessionResult[predictionIndex]),axis=0)

        metricResult = {}
        for i, v in enumerate(self.__model.metrics):
            metricResult[str(v)] = result[i]/sum(batchLen)
        return prediction, metricResult

    def clear(self):
        self.__session.close()
        self.__model.clear()
