import tensorflow as tf
import numpy as np
from NameManger import NameManger

lower = str.lower
nameManger = NameManger()


def lookForLastCheckpoint(model_dir):
    try:
        fr = open(model_dir + 'checkpoint', "r")
    except:
        return None
    f_line = fr.readline()
    start = f_line.find('"')
    end = f_line.rfind('"')
    return f_line[start + 1:end]


def getVariable(shape, name=None, initializer=None, weightDecay=None):
    name = nameManger.getVariableName() if name is None else name
    initializer = 'truncated' if initializer is None else initializer
    if weightDecay is not None:
        regularizer = tf.contrib.layers.l2_regularizer(weightDecay)
    else:
        regularizer = None
    if lower(initializer) == 'random':
        initial_weight = tf.random_normal_initializer(stddev=0.1)
    elif lower(initializer) == 'truncated':
        initial_weight = tf.truncated_normal_initializer(stddev=0.05)
    elif lower(initializer) == 'uniform':
        initial_weight = tf.random_uniform_initializer()
    elif lower(initializer) == 'xavier':
        initial_weight = tf.contrib.layers.xavier_initializer()
    elif lower(initializer) == 'xavier_conv2d':
        initial_weight = tf.contrib.layers.xavier_initializer_conv2d()
    else:
        initial_weight = initializer
    name = nameManger.getVariableName() if name is None else name
    return tf.get_variable(name, shape=shape, initializer=initial_weight, dtype=tf.float32, regularizer=regularizer)


def getWeight(shape, initializer=None, weightDecay=None):
    return getVariable(shape, nameManger.getWeightName(), initializer=initializer)


def getBias(shape, value=None, weightDecay=None):
    initial = tf.constant(0.0 if value is None else value, shape=shape)
    if weightDecay is not None:
        regularizer = tf.contrib.layers.l2_regularizer(weightDecay)
    else:
        regularizer = None
    return tf.Variable(initial, name=nameManger.getBiasName(), dtype=tf.float32)


def getTensorShape(input):
    return input.get_shape().as_list()


def conv2d(input, kernel_size, strides, padding='SAME', initializer=None, weightDecay=None, name=None, with_bias=True,
           groups=1, weightShare=False):
    name = nameManger.getConvName() if name is None else name
    if groups == 1:
        W = getWeight(kernel_size, initializer=initializer, weightDecay=weightDecay)
        conv = tf.nn.conv2d(input, W, strides, padding=padding, name=name)
    else:
        W = getWeight(kernel_size[:-2] + [kernel_size[-2] / groups if weightShare else kernel_size[-2]] + [
            kernel_size[-1] / groups], initializer=initializer, weightDecay=weightDecay)
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=strides, padding=padding)
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=input)
        if weightShare:
            output_groups = [convolve(i, W) for i in input_groups]
        else:
            weight_groups = tf.split(axis=2, num_or_size_splits=groups, value=W)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
        conv = tf.concat(axis=3, values=output_groups, name=name)
    if with_bias:
        tf.nn.bias_add(conv, getBias([kernel_size[3]], weightDecay=weightDecay))
    return conv


def flatten(input):
    op_shape = input.get_shape().as_list()[1:]
    dim = 1
    for value in op_shape:
        dim = dim * value
    return tf.reshape(input, [-1, dim])


def avgPool(input, size, strides, padding='SAME', name=None):
    name = nameManger.getPoolName() if name is None else name
    return tf.nn.avg_pool(input, size, strides, padding, name=name)


def maxPool(input, size, strides, padding='SAME', name=None):
    name = nameManger.getPoolName() if name is None else name
    return tf.nn.max_pool(input, size, strides, padding, name=name)


def globalAvgPool(input, dim=[1, 2], name=None):
    assert input.get_shape().ndims == 4
    name = nameManger.getPoolName() if name is None else name
    return tf.reduce_mean(input, dim, name=name)


def batchNormalization(input, is_training, scale=True):
    return tf.contrib.layers.batch_norm(input, scale=scale, is_training=is_training, updates_collections=None)


def getHiddenLayer(input, size=100, activation='relu', initializer=None, weightDecay=None):
    node_shape = getTensorShape(input)[1:]
    weight = getWeight([node_shape[0], size], initializer, weightDecay=weightDecay)
    bias = getVariable([1, size], name=nameManger.getBiasName(), initializer=initializer, weightDecay=weightDecay)
    output = tf.add(tf.matmul(input, weight), bias)
    if isinstance(activation, int) or isinstance(activation, str):
        output = getActivationFunction(output, activation)
    elif isinstance(activation, list):
        output = getActivationFunction(output, activation)
    return output


def getNHiddenLayers(input, hidden_sizes=None, activation_function_list=None, initializer=None):
    try:
        no_of_layers = len(hidden_sizes)
        len(activation_function_list)
    except:
        no_of_layers = 0
    output = input
    for i in range(no_of_layers):
        with tf.name_scope(nameManger.getHiddenName()):
            output = getHiddenLayer(output, hidden_sizes[i], activation=activation_function_list[i],
                                    initializer=initializer)
    return output


def leakyRelu(node, parameter=0.1):
    shape = getTensorShape(node)[1:]
    const = tf.constant(value=parameter, shape=shape)
    return tf.maximum(node * const, node)


def squash(vector):
    vector += 0.00001
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm)
    vec_squashed = scalar_factor * vector
    return vec_squashed


def getActivationFunction(input, choice="relu", value=None):
    def apply(input, choice, value=None):
        if choice == 0 or lower(str(choice)) == 'none':
            return input
        if choice == 1 or lower(str(choice)) == 'relu':
            return tf.nn.relu(input)
        if choice == 2 or lower(str(choice)) == 'leakyrelu':
            if value == None:
                value = 0.1
            return leakyRelu(input, value)
        if choice == 3 or lower(str(choice)) == 'crelu':
            return tf.nn.crelu(input)
        if choice == 4 or lower(str(choice)) == 'relu6':
            return tf.nn.relu6(input)
        if choice == 5 or lower(str(choice)) == 'elu':
            return tf.nn.elu(input)
        if choice == 6 or lower(str(choice)) == 'sigmoid':
            return tf.nn.sigmoid(input)
        if choice == 7 or lower(str(choice)) == 'tanh':
            return tf.nn.tanh(input)
        if choice == 8 or lower(str(choice)) == 'softplus':
            return tf.nn.softplus(input)
        if choice == 9 or lower(str(choice)) == 'softsign':
            return tf.nn.softsign(input)
        if choice == 10 or lower(str(choice)) == 'softmax':
            return tf.nn.softmax(logits=input)
        if choice == 11 or lower(str(choice)) == 'squash':
            return tf.nn.softmax(logits=input)
        if choice == 11 or lower(str(choice)) == 'pow':
            weight = getVariable(getTensorShape(input)[1:])
            return tf.pow(input, weight)
        if choice == 12 or lower(str(choice)) == 'dropout':
            if value == None:
                value = 0.5
            return tf.nn.dropout(input, value)

    if isinstance(choice, int) or isinstance(choice, str):
        input = apply(input, choice, value)
    elif isinstance(choice, list) or isinstance(choice, tuple):
        for c in choice:
            input = apply(input, c, value)
    return input


def multiFilterBank(input, out_channel={'1': 32, '3': 32, '5': 32}, strides=[1, 1, 1, 1]):
    input_shape = input.get_shape().as_list()[1:]
    con_1x1 = conv2d(input, kernel_size=[1, 1, input_shape[2], out_channel['1']], strides=strides, initializer=None)
    con_3x3 = conv2d(input, kernel_size=[3, 3, input_shape[2], out_channel['3']], strides=strides, initializer=None)
    con_5x5 = conv2d(input, kernel_size=[5, 5, input_shape[2], out_channel['5']], strides=strides, initializer=None)
    pool = maxPool(input, size=[1, 3, 3, 1], strides=strides)
    output = tf.concat([input, con_1x1, con_3x3, con_5x5, pool], 3)
    return output


def getLossAndPrediction(logits, labels, loss_type='softmax', noOfClasses=None):
    if lower(loss_type) == "softmax":
        oneHot = tf.one_hot(labels, noOfClasses,1,0)
        loss = getLoss(logits, oneHot, loss_type)
        probability = tf.nn.softmax(logits, name="softmax")
        prediction = tf.argmax(probability, 1, name='Prediction',)
        return loss, prediction
    elif lower(loss_type) == "hinge":
        oneHot = tf.one_hot(labels, noOfClasses, off_value=-1, on_value=1)
        loss = getLoss(logits, oneHot, loss_type)
        probability = tf.nn.softmax(logits, name="softmax")
        prediction = tf.argmax(probability, 1, name='Prediction')
        return loss, prediction
    elif lower(loss_type) == "binarycrossentropy":
        logits = tf.reshape(logits,[-1])
        loss = getLoss(logits, labels, loss_type)
        prediction = tf.round(logits, name='Prediction')
        return loss, prediction
    elif lower(loss_type) == "mse":
        pass


def getLoss(logits, labels, loss_type='softmax', name=None):
    if lower(loss_type) == "softmax":
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels), name=name)
    elif lower(loss_type) == "hinge":
        cross_entropy = tf.reduce_mean(tf.losses.hinge_loss(logits=logits, labels=labels), name=name)
    elif lower(loss_type) == "huber":
        cross_entropy = tf.reduce_mean(tf.losses.huber_loss(labels=labels, predictions=logits), name=name)
    elif lower(loss_type) == "log":
        cross_entropy = tf.reduce_mean(tf.losses.log_loss(labels=labels, predictions=logits), name=name)
    elif lower(loss_type) == "absolute":
        cross_entropy = tf.reduce_mean(tf.losses.absolute_difference(labels=labels, predictions=logits, name=name))
    elif lower(loss_type) == "mse":
        cross_entropy = tf.losses.mean_squared_error(labels=labels, predictions=logits)
    elif lower(loss_type) == "mpse":
        cross_entropy = tf.losses.mean_pairwise_squared_error(labels=labels, predictions=logits, name=name)
    elif lower(loss_type) == "sigmoid":
        cross_entropy = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(labels, logits, name=name))
    elif lower(loss_type) == "binary_crossentropy":
        cross_entropy = tf.reduce_mean(tf.keras.losses.binary_crossentropy(logits, labels), name=name)
    elif lower(loss_type) == "binarycrossentropy":
        one = tf.constant(1.0)
        labels = tf.cast(labels, dtype=tf.float32)
        cross_entropy = -tf.reduce_mean(
            labels * tf.log(logits) + tf.multiply(tf.subtract(one, labels), tf.log(tf.subtract(one, logits))),
            name=name)
    return cross_entropy


def getRegularizationPanelty(regularizationType='l2', regularizationCoefficient=0.0001, includeBias=True):
    if regularizationType == 'l2':
        weights = tf.trainable_variables()
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=regularizationCoefficient, scope=None)
        l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
        return l2_regularization_penalty
    if regularizationType == 'l1':
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=regularizationCoefficient, scope=None)
        weights = tf.trainable_variables()
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        return regularization_penalty
    if regularizationType == 'elastic_net':
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=regularizationCoefficient[0], scope=None)
        l2_regularizer = tf.contrib.layers.l2_regularizer(scale=regularizationCoefficient[1], scope=None)
        weights = tf.trainable_variables()
        l1_regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
        l2_regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
        regularized_panelty = l1_regularization_penalty + l2_regularization_penalty
        return regularized_panelty


def conv2dDenseBlocktemp(input, is_training, growthRate, strides=[1, 1, 1, 1], layers=5,
                         dropoutRate=0.5, activation='relu', initializer=None, weightDecay=None):
    current = input
    for _ in range(layers):
        with tf.variable_scope(nameManger.getBottleneckName()) as scope:
            tmp = batchNormalization(current, is_training, True)
            tmp = getActivationFunction(tmp, activation)
            tmpShape = getTensorShape(tmp)[1:]
            tmp = conv2d(tmp, [3,3, tmpShape[2],growthRate], strides=strides, initializer=initializer,
                         with_bias=False, weightDecay=weightDecay)
            tmp = tf.nn.dropout(tmp, dropoutRate)
            current = tf.concat([current, tmp], 3)
    return current


def conv2dDenseBlock(input, is_training, growthRate, layers=5, dropoutRate=0.5, activation='relu', initializer=None,
                     weightDecay=None):
    layers_concat = list()
    layers_concat.append(input)

    x = bottleneckLayer(input, is_training, growthRate, activation, dropoutRate, initializer, weightDecay)
    layers_concat.append(x)

    for i in range(layers - 1):
        x = tf.concat(layers_concat, 3)
        x = bottleneckLayer(x, is_training, growthRate, activation, dropoutRate, initializer, weightDecay)
        layers_concat.append(x)
    x = tf.concat(layers_concat, 3)
    return x


def bottleneckLayer(x, isTraining, opFeatures, activation, dropoutRate, initializer=None, weightDecay=None):
    with tf.variable_scope(nameManger.getBottleneckName()) as scope:
        #x = batchNormalization(x, is_training=isTraining)
        #x = getActivationFunction(x, activation)
        #xShape = getTensorShape(x)[1:]
        #x = conv2d(x, [1, 1, xShape[2], 4 * opFeatures], strides=[1, 1, 1, 1], with_bias=False, initializer=initializer,
        #           weightDecay=weightDecay)
        #x = getActivationFunction(x, "dropout", dropoutRate)

        x = batchNormalization(x, is_training=isTraining)
        x = getActivationFunction(x, activation)
        xShape = getTensorShape(x)[1:]
        x = conv2d(x, [3, 3, xShape[2], opFeatures], strides=[1, 1, 1, 1], with_bias=False, initializer=initializer,
                   weightDecay=weightDecay)
        x = getActivationFunction(x, "dropout", dropoutRate)
    return x

def helixBlock(x,isTraining,opfeatures,girth,activation,dropout,initializer,weightDecay,firstBlock=False):

    pass


def residualBlock(input, is_training, output_channel, kernel=3, stride=1, activation="relu", initializer=None,
                  withBN=False,
                  padding_option=0, pad_or_conv=False):
    original_shape = getTensorShape(input)[1:]
    original_input = input

    if withBN:
        input = batchNormalization(input, is_training)
        input = getActivationFunction(input, activation)
        input = conv2d(input, kernel_size=[kernel, kernel, original_shape[2], output_channel],
                       strides=[1, stride, stride, 1], padding='SAME', with_bias=False)
    else:
        input = conv2d(input, kernel_size=[kernel, kernel, original_shape[2], output_channel],
                       strides=[1, stride, stride, 1], initializer=initializer, with_bias=False)

    input = batchNormalization(input, is_training)
    input = getActivationFunction(input, activation)
    input = conv2d(input, kernel_size=[kernel, kernel, output_channel, output_channel],
                   strides=[1, 1, 1, 1], padding='SAME', initializer=initializer, with_bias=False)
    input = batchNormalization(input, is_training)

    if stride != 1:
        original_input = avgPool(original_input, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
    input_shape = getTensorShape(input)[1:]

    if original_shape[2] != input_shape[2] and original_shape[2] < input_shape[2]:
        if pad_or_conv == True:
            original_input = conv2d(original_input, [1, 1, original_shape[2], input_shape[2]],
                                    [1, 1, 1, 1], with_bias=False)
        else:
            original_input = tf.pad(original_input,
                                    [[padding_option, padding_option], [padding_option, padding_option],
                                     [padding_option, padding_option],
                                     [(input_shape[2] - original_shape[2]) // 2,
                                      (input_shape[2] - original_shape[2]) // 2]])
    elif original_shape[2] != input_shape[2] and original_shape[2] > input_shape[2]:
        original_input = conv2d(original_input, [1, 1, original_shape[2], input_shape[2]],
                                [1, 1, 1, 1], with_bias=False)

    return getActivationFunction(input + original_input, activation)


def residualBottleneckBlock(input, is_training, output_channel, kernel=3, stride=1, withBN=False,
                            padding_option=0, pad_or_conv=False, initializer=None, activation="relu"):
    original_shape = input.get_shape().as_list()[1:]
    original_input = input

    if withBN:
        input = batchNormalization(input, is_training)
        input = getActivationFunction(input, activation)
        input = conv2d(input, kernel_size=[1, 1, original_shape[2], output_channel / 4],
                       strides=[1, stride, stride, 1], padding='SAME', initializer=initializer, with_bias=False)
    else:
        input = conv2d(input, kernel_size=[1, 1, original_shape[2], output_channel / 4],
                       strides=[1, stride, stride, 1], padding='SAME', initializer=initializer, with_bias=False)

    input = batchNormalization(input, is_training)
    input = getActivationFunction(input, activation)
    input = conv2d(input, kernel_size=[kernel, kernel, output_channel / 4, output_channel / 4],
                   strides=[1, 1, 1, 1], padding='SAME', initializer=initializer, with_bias=False)

    input = batchNormalization(input, is_training)
    input = getActivationFunction(input, activation)
    input = conv2d(input, kernel_size=[kernel, kernel, output_channel / 4, output_channel],
                   strides=[1, 1, 1, 1], padding='SAME', initializer=initializer, with_bias=False)

    if stride != 1:
        original_input = maxPool(original_input, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
    input_shape = getTensorShape(input)[1:]

    if original_shape[2] != input_shape[2] and original_shape[2] < input_shape[2]:
        if pad_or_conv == True:
            original_input = conv2d(original_input, [1, 1, original_shape[2], input_shape[2]],
                                    [1, 1, 1, 1], with_bias=False)
        else:
            original_input = tf.pad(original_input, [[padding_option, padding_option], [padding_option, padding_option],
                                                     [padding_option, padding_option],
                                                     [(input_shape[2] - original_shape[2]) // 2,
                                                      (input_shape[2] - original_shape[2]) // 2]])
    elif original_shape[2] != input_shape[2] and original_shape[2] > input_shape[2]:
        original_input = conv2d(original_input, [1, 1, original_shape[2], input_shape[2]],
                                [1, 1, 1, 1], with_bias=False)

    return getActivationFunction(input + original_input, activation)


def residualWideBlock(input, is_training, output_channel, kernel=3, stride=1, withBN=False,
                      padding_option=0, pad_or_conv=False, dropout_rate=0.5, initializer=None, activation="relu"):
    original_shape = input.get_shape().as_list()[1:]
    original_input = input
    if withBN:
        input = batchNormalization(input, is_training)
        input = getActivationFunction(input, activation)
        input = conv2d(input, kernel_size=[kernel, kernel, original_shape[2], output_channel],
                       strides=[1, stride, stride, 1], padding='SAME', initializer=initializer, with_bias=False)
    else:
        input = conv2d(input, kernel_size=[kernel, kernel, original_shape[2], output_channel],
                       strides=[1, stride, stride, 1], padding='SAME', initializer=initializer, with_bias=False)

    input = getActivationFunction(input, "dropout", dropout_rate)
    input = batchNormalization(input, is_training)
    input = getActivationFunction(input, activation)
    input = conv2d(input, kernel_size=[kernel, kernel, output_channel, output_channel],
                   strides=[1, 1, 1, 1], padding='SAME', initializer=initializer, with_bias=False)

    if stride != 1:
        original_input = maxPool(original_input, [1, stride, stride, 1], [1, stride, stride, 1], 'VALID')
    input_shape = getTensorShape(input)[1:]

    if original_shape[2] != input_shape[2] and original_shape[2] < input_shape[2]:
        if pad_or_conv == True:
            original_input = conv2d(original_input, [1, 1, original_shape[2], input_shape[2]],
                                    [1, 1, 1, 1], initializer=initializer, with_bias=False)
        else:
            original_input = tf.pad(original_input,
                                    [[padding_option, padding_option], [padding_option, padding_option],
                                     [padding_option, padding_option],
                                     [(input_shape[2] - original_shape[2]) // 2,
                                      (input_shape[2] - original_shape[2]) // 2]])
    elif original_shape[2] != input_shape[2] and original_shape[2] > input_shape[2]:
        original_input = conv2d(original_input, [1, 1, original_shape[2], input_shape[2]],
                                [1, 1, 1, 1], initializer=initializer, with_bias=False)

    return getActivationFunction(input + original_input)


'''

def inception_naive_block(input, name, is_training, out_channel={'1': 32, '3': 32, '5': 32}, strides=[1, 1, 1, 1]):
    input_shape = input.get_shape().as_list()[1:]
    con_1x1 = conv2d(input, name + '_conv1X1', kernel_size=[1, 1, input_shape[2], out_channel['1']], strides=strides,
                     initial='xavier')
    con_3x3 = conv2d(input, name + '_conv3X3', kernel_size=[3, 3, input_shape[2], out_channel['3']], strides=strides,
                     initial='xavier')
    con_5x5 = conv2d(input, name + '_conv5X5', kernel_size=[5, 5, input_shape[2], out_channel['5']], strides=strides,
                     initial='xavier')
    pool = max_pool(input, size=[1, 3, 3, 1], strides=strides)
    output = tf.concat([con_1x1, con_3x3, con_5x5, pool], 3)
    return output


def inception_v2_block(input, name, is_training, strides=[1, 1, 1, 1], out_channel={'1': 32, '3': 32, '5': 32},
                       reduced_out_channel={'3': 32, '5': 32, 'p': 32}):
    input_shape = input.get_shape().as_list()[1:]
    con_1x1 = conv2d(input, name + '_conv1X1', kernel_size=[1, 1, input_shape[2], out_channel['1']], strides=strides,
                     initial='xavier')
    for_3x3_con_1x1 = conv2d(input, name + '_conv1X1_3X3', kernel_size=[1, 1, input_shape[2], reduced_out_channel['3']],
                             strides=strides, initial='xavier')
    con_3x3 = conv2d(for_3x3_con_1x1, name + '_conv3X3', kernel_size=[3, 3, reduced_out_channel['3'], out_channel['3']],
                     strides=strides,
                     initial='xavier')
    for_5x5_con_1x1 = conv2d(input, name + '_conv1X1_5X5', kernel_size=[1, 1, input_shape[2], reduced_out_channel['5']],
                             strides=strides, initial='xavier')
    con_5x5 = conv2d(for_5x5_con_1x1, name + '_conv5X5', kernel_size=[5, 5, reduced_out_channel['5'], out_channel['5']],
                     strides=strides,
                     initial='xavier')
    pool = max_pool(input, size=[1, 3, 3, 1], strides=strides)
    max_1x1 = conv2d(pool, name + '_conv1X1_max', kernel_size=[1, 1, input_shape[2], reduced_out_channel['p']],
                     strides=strides,
                     initial='xavier')
    output = tf.concat([con_1x1, con_3x3, con_5x5, max_1x1], 3)
    return output


def inception_v3_block(self, input, name, is_training, strides=[1, 1, 1, 1],
                       out_channel={'1': 32, '3_1': 32, '3_2': 32},
                       reduced_out_channel={'3_1': 32, '3_2': 32, 'p': 32}):
    input_shape = input.get_shape().as_list()[1:]
    con_1x1 = self.conv2d(input, name + '_conv1X1', kernel_size=[1, 1, input_shape[2], out_channel['1']],
                          strides=strides,
                          initial='xavier')
    for_3x3_con_1x1 = self.conv2d(input, name + '_conv1X1_3X3_1',
                                  kernel_size=[1, 1, input_shape[2], reduced_out_channel['3_1']],
                                  strides=strides, initial='xavier')
    con_3x3 = self.conv2d(for_3x3_con_1x1, name + '_conv_3X3_1',
                          kernel_size=[3, 3, reduced_out_channel['3_1'], out_channel['3_1']],
                          strides=strides,
                          initial='xavier')
    for_3x3_2_con_1x1 = self.conv2d(input, name + '_conv1X1_3X3_2',
                                    kernel_size=[1, 1, input_shape[2], reduced_out_channel['3_2']],
                                    strides=strides, initial='xavier')
    con_3x3_2 = self.conv2d(for_3x3_2_con_1x1, name + '_conv3X3_2',
                            kernel_size=[3, 3, reduced_out_channel['3_2'], out_channel['3_2']],
                            strides=strides,
                            initial='xavier')
    con_3x3_2 = self.conv2d(con_3x3_2, name + '_conv3X3_3', kernel_size=[3, 3, out_channel['3_2'], out_channel['3_2']],
                            strides=strides,
                            initial='xavier')
    pool = self.max_pool(input, size=[1, 3, 3, 1], strides=strides)
    max_1x1 = self.conv2d(pool, name + '_conv1X1max', kernel_size=[1, 1, input_shape[2], reduced_out_channel['p']],
                          strides=strides,
                          initial='xavier')
    output = tf.concat([con_1x1, con_3x3, con_3x3_2, max_1x1], 3)
    return output
'''


def getNoOfParameters(name=None):
    total_parameters = 0
    for variable in tf.trainable_variables():
        if name is None or name in lower(variable.name):
            shape = variable.get_shape()
            variable_parameters = 1

            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
    return total_parameters


def pre_process(images, is_training):
    def pre_process_image(image, is_training):
        # This function takes a single image as input,
        # and a boolean whether to build the training or testing graph.
        img_size_cropped = 32
        if is_training:
            # For training, add the following to the TensorFlow graph.

            # Randomly crop the input image.
            image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, 3])

            # Randomly flip the image horizontally.
            image = tf.image.random_flip_left_right(image)

            # Randomly adjust hue, contrast and saturation.
            image = tf.image.random_hue(image, max_delta=0.05)
            image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

            # Some of these functions may overflow and result in pixel
            # values beyond the [0, 1] range. It is unclear from the
            # documentation of TensorFlow 0.10.0rc0 whether this is
            # intended. A simple solution is to limit the range.

            # Limit the image pixels between [0, 1] in case of overflow.
            image = tf.minimum(image, 1.0)
            image = tf.maximum(image, 0.0)
        else:
            # For training, add the following to the TensorFlow graph.

            # Crop the input image around the centre so it is the same
            # size as images that are randomly cropped during training.
            image = tf.image.resize_image_with_crop_or_pad(image,
                                                           target_height=img_size_cropped,
                                                           target_width=img_size_cropped)
        return image

    images = tf.map_fn(lambda image: pre_process_image(image, is_training), images)
    return images


def getMetric(pred, target, type):
    if lower(type) == 'accuracy':
        return getAccuracy(pred, target)


def getAccuracy(pred, target):
    target = tf.reshape(target, [-1])
    pred = tf.reshape(pred, [-1])
    correct_prediction = tf.equal(tf.cast(pred, dtype=tf.int32), target)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='Accuracy')


def getRecall(pred, target):
    tf.metrics.recall(
        target,
        pred,
        weights=None,
        metrics_collections=None,
        updates_collections=None,
        name=None
    )


def perceptual_distance(pred, target):
    rmean = (target[:, :, :, 0] + pred[:, :, :, 0]) / 2
    r = target[:, :, :, 0] - pred[:, :, :, 0]
    g = target[:, :, :, 1] - pred[:, :, :, 1]
    b = target[:, :, :, 2] - pred[:, :, :, 2]
    c512 = tf.constant(512)
    c256 = tf.constant(256)
    c4 = tf.constant(4)
    c767 = tf.constant(767)
    return tf.reduce_mean(tf.sqrt(tf.add(tf.div(tf.multiply(tf.add(c512, rmean), tf.square(r)), c256),
                                         tf.multiply(c4, tf.square(g)) + tf.div(
                                             tf.multiply((c767 - rmean), tf.square(b)), c256))))


def imageNormalize(input):
    inputGroups = tf.split(axis=3, num_or_size_splits=3, value=input)
    outputGroups = []
    mean = [103.062623801, 115.902882574, 123.151630838, ]
    for i in range(3):
        outputGroups.append(tf.subtract(inputGroups[i], tf.constant(mean[i], shape=getTensorShape(inputGroups[i])[1:])))
    outputGroups = tf.concat(outputGroups, axis=3)
    return tf.div(outputGroups, tf.constant(255.0))
