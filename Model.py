import tensorflow as tf
import ops

lower = str.lower


class Network:
    imageShape = None
    noOfClasses = None
    regularizationType = None
    regularizationCoefficient = 0.001
    hiddenUnits = 4096
    fcActivation = "relu"
    convActivation = "relu"
    convWeightDecay = 0.00004
    fcWeightDecay = 0.00004
    __lossType = "softmax"
    inputs = {}
    outputs = {}
    metrics = []

    def build(self):
        tf.reset_default_graph()
        with tf.name_scope('Inputs'):
            self.__x = tf.placeholder(dtype=tf.float32, shape=[None] + self.imageShape, name="input")
            self.__y = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
            self.__lr = tf.placeholder("float", shape=[], name='learningRate')
            self.__isTrain = tf.placeholder(tf.bool, shape=[], name='isTraining')
            self.inputs = {'x': self.__x, 'y': self.__y, 'learningRate': self.__lr, 'isTraining': self.__isTrain}

        self.__logits = self.__getModel(self.__x, self.__isTrain)

        with tf.name_scope('Output'):
            self.__loss, self.__prediction = ops.getLossAndPrediction(self.__logits, self.__y, self.__lossType,
                                                                      self.noOfClasses)
            if self.regularizationType is not None:
                panelty = ops.getRegularizationPanelty(self.regularizationType, self.regularizationCoefficient)
                self.__loss = tf.add(self.__loss, panelty, name='loss')
            tf.summary.scalar("Loss", self.__loss)

            for m in self.metrics:
                self.outputs[str(m)] = ops.getMetric(self.__prediction, self.__y, m)
                tf.summary.scalar(str(m), self.outputs[str(m)])

            self.outputs['loss'] = self.__loss
            self.outputs['prediction'] = self.__prediction
            self.metrics.append('loss')

    def __getModel(self, x, is_training):
        x_shape = x.get_shape().as_list()[1:]
        kernel = {
            'c1_1': [3, 3, x_shape[2], 64], 'c1_2': [3, 3, 64, 64],
            'c2_1': [3, 3, 64, 128], 'c2_2': [3, 3, 128, 128],
            'c3_1': [3, 3, 128, 256], 'c3_2': [3, 3, 256, 256],
            'c3_3': [3, 3, 256, 256],
            'c4_1': [3, 3, 256, 512], 'c4_2': [3, 3, 512, 512],
            'c4_3': [3, 3, 512, 512],
            'c5_1': [3, 3, 512, 512], 'c5_2': [3, 3, 512, 512],
            'c5_3': [3, 3, 512, 512]}
        strides = {'c': [1, 1, 1, 1], 'p': [1, 2, 2, 1]}
        pool_win_size = [1, 2, 2, 1]
        conv = x
        with tf.variable_scope('Conv_1') as scope:
            conv = ops.conv2d(conv, kernel['c1_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c1_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_2') as scope:
            conv = ops.conv2d(conv, kernel['c2_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c2_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_3') as scope:
            conv = ops.conv2d(conv, kernel['c3_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c3_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c3_3'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])

        with tf.variable_scope('Conv_4') as scope:
            conv = ops.conv2d(conv, kernel['c4_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c4_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c4_3'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Flatten_layer') as scope:
            conv = ops.flatten(conv)
        with tf.variable_scope('Hidden_layer_1') as scope:
            conv = ops.getHiddenLayer(conv, self.hiddenUnits, activation=self.fcActivation, initializer="truncated",
                                      weightDecay=self.fcWeightDecay)
        with tf.variable_scope('Hidden_layer_2') as scope:
            conv = ops.getHiddenLayer(conv, self.hiddenUnits, activation=self.fcActivation, initializer="truncated",
                                      weightDecay=self.fcWeightDecay)
        with tf.variable_scope('Output_layer') as scope:
            conv = ops.getHiddenLayer(conv, self.noOfClasses, activation=self.fcActivation, initializer="truncated",
                                      weightDecay=self.fcWeightDecay)
        return conv

    def getNumberOfParameters(self, type=None):
        return ops.getNoOfParameters(type)

    def addMetrics(self, type):
        self.metrics.append(type)

    def clear(self):
        tf.reset_default_graph()

class Vgg16:
    imageShape = None
    noOfClasses = None
    regularizationType = None
    regularizationCoefficient = 0.001
    hiddenUnits = 4096
    fcActivation = "relu"
    convActivation = "relu"
    convWeightDecay = 0.00004
    fcWeightDecay = 0.00004
    __lossType = "softmax"
    inputs = {}
    outputs = {}
    metrics = []

    def build(self):
        tf.reset_default_graph()
        with tf.name_scope('Inputs'):
            self.__x = tf.placeholder(dtype=tf.float32, shape=[None] + self.imageShape, name="input")
            self.__y = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
            self.__lr = tf.placeholder("float", shape=[], name='learningRate')
            self.__isTrain = tf.placeholder(tf.bool, shape=[], name='isTraining')
            self.inputs = {'x': self.__x, 'y': self.__y, 'learningRate': self.__lr, 'isTraining': self.__isTrain}

        self.__logits = self.__getModel(self.__x, self.__isTrain)

        with tf.name_scope('Output'):
            self.__loss, self.__prediction = ops.getLossAndPrediction(self.__logits, self.__y, self.__lossType,
                                                                      self.noOfClasses)
            if self.regularizationType is not None:
                panelty = ops.getRegularizationPanelty(self.regularizationType, self.regularizationCoefficient)
                self.__loss = tf.add(self.__loss, panelty, name='loss')
            tf.summary.scalar("Loss", self.__loss)

            for m in self.metrics:
                self.outputs[str(m)] = ops.getMetric(self.__prediction, self.__y, m)
                tf.summary.scalar(str(m), self.outputs[str(m)])

            self.outputs['loss'] = self.__loss
            self.outputs['prediction'] = self.__prediction
            self.metrics.append('loss')

    def __getModel(self, x, is_training):
        x_shape = x.get_shape().as_list()[1:]
        kernel = {
            'c1_1': [3, 3, x_shape[2], 64], 'c1_2': [3, 3, 64, 64],
            'c2_1': [3, 3, 64, 128], 'c2_2': [3, 3, 128, 128],
            'c3_1': [3, 3, 128, 256], 'c3_2': [3, 3, 256, 256],
            'c3_3': [3, 3, 256, 256],
            'c4_1': [3, 3, 256, 512], 'c4_2': [3, 3, 512, 512],
            'c4_3': [3, 3, 512, 512],
            'c5_1': [3, 3, 512, 512], 'c5_2': [3, 3, 512, 512],
            'c5_3': [3, 3, 512, 512]}
        strides = {'c': [1, 1, 1, 1], 'p': [1, 2, 2, 1]}
        pool_win_size = [1, 2, 2, 1]
        conv = x
        with tf.variable_scope('Conv_1') as scope:
            conv = ops.conv2d(conv, kernel['c1_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c1_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_2') as scope:
            conv = ops.conv2d(conv, kernel['c2_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c2_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_3') as scope:
            conv = ops.conv2d(conv, kernel['c3_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c3_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c3_3'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])

        with tf.variable_scope('Conv_4') as scope:
            conv = ops.conv2d(conv, kernel['c4_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c4_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c4_3'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
            
        with tf.variable_scope('Conv_5') as scope:
            conv = ops.conv2d(conv, kernel['c5_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c5_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c5_3'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
            
        with tf.variable_scope('Flatten_layer') as scope:
            conv = ops.flatten(conv)
        with tf.variable_scope('Hidden_layer_1') as scope:
            conv = ops.getHiddenLayer(conv, self.hiddenUnits, activation=self.fcActivation, initializer="truncated",
                                      weightDecay=self.fcWeightDecay)
        with tf.variable_scope('Hidden_layer_2') as scope:
            conv = ops.getHiddenLayer(conv, self.hiddenUnits, activation=self.fcActivation, initializer="truncated",
                                      weightDecay=self.fcWeightDecay)
        with tf.variable_scope('Output_layer') as scope:
            conv = ops.getHiddenLayer(conv, self.noOfClasses, activation=self.fcActivation, initializer="truncated",
                                      weightDecay=self.fcWeightDecay)
        return conv

    def getNumberOfParameters(self, type=None):
        return ops.getNoOfParameters(type)

    def addMetrics(self, type):
        self.metrics.append(type)

    def clear(self):
        tf.reset_default_graph()

class Vgg19:
    imageShape = None
    noOfClasses = None
    regularizationType = None
    regularizationCoefficient = 0.001
    hiddenUnits = 4096
    fcActivation = "relu"
    convActivation = "relu"
    convWeightDecay = 0.00004
    fcWeightDecay = 0.00004
    __lossType = "softmax"
    inputs = {}
    outputs = {}
    metrics = []

    def build(self):
        tf.reset_default_graph()
        with tf.name_scope('Inputs'):
            self.__x = tf.placeholder(dtype=tf.float32, shape=[None] + self.imageShape, name="input")
            self.__y = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
            self.__lr = tf.placeholder("float", shape=[], name='learningRate')
            self.__isTrain = tf.placeholder(tf.bool, shape=[], name='isTraining')
            self.inputs = {'x': self.__x, 'y': self.__y, 'learningRate': self.__lr, 'isTraining': self.__isTrain}

        self.__logits = self.__getModel(self.__x, self.__isTrain)

        with tf.name_scope('Output'):

            self.__loss, self.__prediction = ops.getLossAndPrediction(self.__logits, self.__y, self.__lossType,
                                                                      self.noOfClasses)
            if self.regularizationType is not None:
                panelty = ops.getRegularizationPanelty(self.regularizationType, self.regularizationCoefficient)
                self.__loss = tf.add(self.__loss, panelty, name='loss')
            tf.summary.scalar("Loss", self.__loss)

            for m in self.metrics:
                self.outputs[str(m)] = ops.getMetric(self.__prediction, self.__y, m)
                tf.summary.scalar(str(m), self.outputs[str(m)])

            self.outputs['loss'] = self.__loss
            self.outputs['prediction'] = self.__prediction
            self.metrics.append('loss')

    def __getModel(self, x, is_training):
        x_shape = x.get_shape().as_list()[1:]
        kernel = {'c1_1': [3, 3, x_shape[2], 64], 'c1_2': [3, 3, 64, 64],
                  'c2_1': [3, 3, 64, 128], 'c2_2': [3, 3, 128, 128],
                  'c3_1': [3, 3, 128, 256], 'c3_2': [3, 3, 256, 256],
                  'c3_3': [3, 3, 256, 256], 'c3_4': [3, 3, 256, 256],
                  'c4_1': [3, 3, 256, 512], 'c4_2': [3, 3, 512, 512],
                  'c4_3': [3, 3, 512, 512], 'c4_4': [3, 3, 512, 512],
                  'c5_1': [3, 3, 512, 512], 'c5_2': [3, 3, 512, 512],
                  'c5_3': [3, 3, 512, 512], 'c5_4': [3, 3, 512, 512]}
        strides = {'c': [1, 1, 1, 1], 'p': [1, 2, 2, 1]}
        pool_win_size = [1, 2, 2, 1]
        conv = x

        with tf.variable_scope('Conv_1') as scope:
            conv = ops.conv2d(conv, kernel['c1_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c1_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_2') as scope:
            conv = ops.conv2d(conv, kernel['c2_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c2_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_3') as scope:
            conv = ops.conv2d(conv, kernel['c3_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c3_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c3_3'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c3_4'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_4') as scope:
            conv = ops.conv2d(conv, kernel['c4_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c4_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c4_3'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c4_4'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Conv_5') as scope:
            conv = ops.conv2d(conv, kernel['c5_1'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c5_2'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c5_3'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.conv2d(conv, kernel['c5_4'], strides['c'], 'SAME', initializer="truncated",
                              weightDecay=self.convWeightDecay)
            conv = ops.getActivationFunction(conv, self.convActivation)
            conv = ops.maxPool(conv, pool_win_size, strides['p'])
        with tf.variable_scope('Flatten_layer') as scope:
            conv = ops.flatten(conv)
        with tf.variable_scope('Hidden_layer_1') as scope:
            conv = ops.getHiddenLayer(conv, self.hiddenUnits, activation=self.fcActivation, initializer="truncated",
                                      weightDecay=self.fcWeightDecay)
        with tf.variable_scope('Hidden_layer_2') as scope:
            conv = ops.getHiddenLayer(conv, self.hiddenUnits, activation=self.fcActivation, initializer="truncated",
                                      weightDecay=self.fcWeightDecay)
        with tf.variable_scope('Output_layer') as scope:
            conv = ops.getHiddenLayer(conv, self.noOfClasses, activation=self.fcActivation, initializer="truncated",
                                      weightDecay=self.fcWeightDecay)
        return conv

    def getNumberOfParameters(self, type=None):
        return ops.getNoOfParameters(type)

    def addMetrics(self, type):
        self.metrics.append(type)

    def clear(self):
        tf.reset_default_graph()
