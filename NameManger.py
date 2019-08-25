class NameManger:
    def __init__(self):
        self.__convCount = 0
        self.__weightCount = 0
        self.__biasCount = 0
        self.__variableCount = 0
        self.__blockCount = 0
        self.__bottleneckCount = 0
        self.__poolCount = 0
        self.__inputName = 0
        self.__outputName = 0
        self.__name = 0
        self.__placeHolderCount = 0
        self.__batchNormCount = 0

    def getConvName(self):
        self.__convCount = self.__convCount + 1
        return "Conv_" + str(self.__convCount)

    def getHiddenName(self):
        self.__convCount = self.__convCount + 1
        return "HiddenLayer_" + str(self.__convCount)

    def getWeightName(self):
        self.__weightCount = self.__weightCount + 1
        return "Weight_" + str(self.__weightCount)

    def getBiasName(self):
        self.__biasCount = self.__biasCount + 1
        return "Bias_" + str(self.__biasCount)

    def getVariableName(self):
        self.__variableCount = self.__variableCount + 1
        return "Variable_" + str(self.__variableCount)

    def getBlockName(self):
        self.__blockCount = self.__blockCount + 1
        return "Block_" + str(self.__blockCount)

    def getBottleneckName(self):
        self.__bottleneckCount = self.__bottleneckCount + 1
        return "Bottleneck_" + str(self.__bottleneckCount)

    def getPoolName(self):
        self.__poolCount = self.__poolCount + 1
        return "Pool_" + str(self.__poolCount)

    def getInputName(self):
        self.__inputName = self.__inputName + 1
        return "Input_" + str(self.__inputName)

    def getOutputName(self):
        self.__outputName = self.__outputName + 1
        return "Output_" + str(self.__outputName)

    def getName(self):
        self.__name = self.__name + 1
        return 'Node_' + str(self.__name)

    def getPlaceHolderName(self):
        self.__placeHolderCount = self.__placeHolderCount + 1
        return 'PlaceHolder_' + str(self.__placeHolderCount)

    def getBatchNormName(self):
        self.__batchNormCount = self.__batchNormCount + 1
        return 'BatchNorm_' + str(self.__batchNormCount)