import numpy as np
import utils
import os, random
import h5py

try:
    from ImageOps import ImageOps

    ImageOps = ImageOps()
except:
    ImageOps = None


def readMnist(path, one_hot=False):
    import gzip, os, sys, time
    def extract_data(filename, num_images):
        IMAGE_SIZE = 28
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            # data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
            data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
            return data

    def extract_labels(filename, num_images):
        """Extract the labels into a vector of int64 label IDs."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels.reshape(num_images, 1)

    train_x = extract_data(path + 'train-images-idx3-ubyte.gz', 60000)
    train_y = extract_labels(path + 'train-labels-idx1-ubyte.gz', 60000)
    test_x = extract_data(path + 't10k-images-idx3-ubyte.gz', 10000)
    test_y = extract_labels(path + 't10k-labels-idx1-ubyte.gz', 10000)
    if one_hot == True:
        train_y = utils.convert_to_onehot(train_y, 10)
        test_y = utils.convert_to_onehot(test_y, 10)
    return {'x': train_x.reshape([-1, 28, 28, 1]), 'y': train_y.reshape([-1])}, {'x': test_x.reshape([-1, 28, 28, 1]),
                                                                                 'y': test_y.reshape([-1])}


def readCifar10(path, one_hot=False):
    text_labels = utils.load_model(path + 'batches.meta')['label_names']
    for i in range(5):
        if os.name == "nt":
            data = utils.load_model(path + 'data_batch_' + str(i + 1))
        else:
            data = utils.load_encoding_model(path + 'data_batch_' + str(i + 1), encode='bytes')
        if i == 0:
            train_x = data['data' if os.name == "nt" else b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1,
                                                                                                                    2).reshape(
                -1, 32 * 32 * 3)
            train_y = np.array(data['labels' if os.name == "nt" else b'labels']).reshape(10000, 1)
            continue
        train_x = np.vstack((train_x,
                             data['data' if os.name == "nt" else b'data'].reshape((-1, 3, 32, 32)).swapaxes(1,
                                                                                                            3).swapaxes(
                                 1, 2).reshape(-1,
                                               32 * 32 * 3)))
        train_y = np.vstack((train_y, np.array(data['labels' if os.name == "nt" else b'labels']).reshape(10000, 1)))
    data = utils.load_encoding_model(path + 'test_batch', encode='bytes')
    test_x = data['data' if os.name == "nt" else b'data'].reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1,
                                                                                                           2).reshape(
        -1, 32 * 32 * 3)
    test_y = np.array(data['labels' if os.name == "nt" else b'labels']).reshape(10000, 1)
    if one_hot == True:
        train_y = utils.convert_to_onehot(train_y, 10)
        test_y = utils.convert_to_onehot(test_y, 10)
    else:
        test_y = np.reshape(test_y,(-1))
        train_y = np.reshape(train_y, (-1))
    train_x = np.reshape(train_x, (-1,32,32,3))
    test_x = np.reshape(test_x, (-1,32,32,3))
    return {'x': train_x, 'y': train_y}, {'x': test_x, 'y': test_y}, {'text_labels': text_labels}


def readIris(classes=3, one_hot=False):
    from sklearn.datasets import load_iris
    iris = load_iris()
    x = iris['data']
    y = iris['target']
    train_x1 = x[0:40, :]
    train_x2 = x[50:90, :]
    train_x3 = x[100:140, :]
    train_y1 = y[0:40].reshape(40, 1)
    train_y2 = y[50:90].reshape(40, 1)
    train_y3 = y[100:140].reshape(40, 1)
    test_x1 = x[40:50, :]
    test_x2 = x[90:100, :]
    test_x3 = x[140:150, :]
    test_y1 = y[40:50].reshape(10, 1)
    test_y2 = y[90:100].reshape(10, 1)
    test_y3 = y[140:150].reshape(10, 1)
    train_x = np.vstack((train_x1, train_x2))
    train_y = np.vstack((train_y1, train_y2))
    test_x = np.vstack((test_x1, test_x2))
    test_y = np.vstack((test_y1, test_y2))
    if (classes == 3):
        train_x = np.vstack((train_x, train_x3))
        train_y = np.vstack((train_y, train_y3))
        test_x = np.vstack((test_x, test_x3))
        test_y = np.vstack((test_y, test_y3))
    if one_hot == True:
        train_y = utils.convert_to_onehot(train_y, classes)
        test_y = utils.convert_to_onehot(test_y, classes)
    return {'x': train_x, 'y': train_y.reshape([-1])}, {'x': test_x, 'y': test_y.reshape([-1])}


def readCifar100(path="/media/batman/ent/datasets/cifar-100-python", oneHot=False, labelType="fine"):
    textLabels = utils.load_model(os.path.join(path, 'meta'))['fine_label_names']

    if os.name == "nt":
        train = utils.load_model(os.path.join(path, 'train'))
        test = utils.load_model(os.path.join(path, 'test'))
    else:
        train = utils.load_encoding_model(os.path.join(path, 'train'), encode='bytes')
        test = utils.load_encoding_model(os.path.join(path, 'test'), encode='bytes')
    trainFilenames = []
    trainDict = list(train.keys())
    testDict = list(test.keys())
    for f in train[trainDict[0]]:
        f = str(f)
        trainFilenames.append((f.split('_s_')[0]).replace("_", " "))
    testFilenames = []
    for f in test[testDict[0]]:
        f = str(f)
        testFilenames.append((f.split('_s_')[0]).replace("_", " "))

    if labelType == "fine":
        trainY = np.array(train[trainDict[2]])
        testY = np.array(test[testDict[2]])
    else:
        trainY = np.array(train[trainDict[3]])
        testY = np.array(test[testDict[3]])

    trainX = np.array(train[trainDict[4]]).reshape((-1, 32, 32, 3))
    testX = np.array(test[testDict[4]]).reshape((-1, 32, 32, 3))

    if oneHot == True:
        trainY = utils.convert_to_onehot(trainY, 10 if labelType == "fine" else 100)
        testX = utils.convert_to_onehot(testY, 10 if labelType == "fine" else 100)
    return {'x': trainX, 'y': trainY, "filenames": trainFilenames}, {'x': testX, 'y': testY,
                                                                     "filenames": testFilenames}, textLabels


class ClassificationImageGenerator:
    def __init__(self, path, batchSize=64, batchType='random', imageSize=32):
        self.__path = path
        self.batchSize = batchSize
        self.batchType = batchType
        self.imageSize = imageSize
        self.__classes = {}
        self.__sequentialIndex = None
        self.__lastPointer = 0
        return

    def init(self):
        metaDataPath = os.path.join(self.__path, 'metadata.txt')
        metaData = None
        if os.path.exists(metaDataPath):
            metaDataReader = open(metaDataPath, 'r')
            metaData = [line.replace('\n', '') for line in metaDataReader.readlines()]
        classIndex = [dir for dir in os.listdir(self.__path) if os.path.isdir(os.path.join(self.__path, dir))]
        for i in classIndex:
            currentClassRoot = os.path.join(self.__path, i)
            currentFilesPath = [os.path.join(currentClassRoot, file) for file in os.listdir(currentClassRoot) if
                                os.path.isfile(os.path.join(currentClassRoot, file))]
            self.__classes[i] = {
                'name': metaData[int(i)] if metaData is not None else i,
                'filesPathList': currentFilesPath,
                'classRootPath': currentClassRoot
            }
        if self.__classes.keys() == 0:
            print('Nothing To Build')
        return

    def batchGenerator(self):
        classCount = len(self.__classes.keys())
        resultX = []
        resultY = []
        if self.batchType.lower() == 'random':
            i = 0
            classIndex = self.__lastPointer % classCount
            while i < self.batchSize:
                itemIndex = random.sample(range(0, len(self.__classes[str(classIndex)]['filesPathList']), ), 1)
                resultX.append(self.__classes[str(classIndex)]['filesPathList'][itemIndex[0]])
                resultY.append(classIndex)
                classIndex = (classIndex + 1) % classCount
                i = i + 1
            self.__lastPointer = classIndex
        elif self.batchType.lower() == 'sequential':
            if self.__sequentialIndex is None:
                self.__sequentialIndex = {}
                for key in self.__classes.keys():
                    self.__sequentialIndex[key] = 0
            i = 0
            classIndex = self.__lastPointer % classCount
            while i < self.batchSize:
                itemIndex = self.__sequentialIndex[str(classIndex)]
                self.__sequentialIndex[str(classIndex)] = (self.__sequentialIndex[str(classIndex)] + 1) % \
                                                          len(self.__classes[str(classIndex)]['filesPathList'])
                resultX.append(self.__classes[str(classIndex)]['filesPathList'][itemIndex])
                resultY.append(classIndex)
                classIndex = (classIndex + 1) % classCount
                i = i + 1
            self.__lastPointer = classIndex
        return resultX, resultY

    def batchProcessor(self, pathX, Y):
        resultX = []
        resultY = []
        for i in range(len(pathX)):
            try:
                x = ImageOps.readImage(pathX[i])
                x = ImageOps.resizeImageWithAspectRatio(x, self.imageSize)
                x = np.reshape(x, (-1))
                if x is None or len(x) != self.imageSize * self.imageSize * 3:
                    raise Exception
                resultX.append(np.array(x)/255.0)
                resultY.append(Y[i])
            except:
                print("file removed:", pathX[i])
                #os.remove(pathX[i])
        return np.concatenate(resultX, axis=0).reshape((-1, self.imageSize, self.imageSize, 3)), np.array(
            resultY).reshape((-1))


class CatzGenerator:
    def __init__(self, path, batchSize=64):
        self.__path = path
        self.batchSize = batchSize

    def init(self):
        self.trainDirectory = [os.path.join(self.__path, dir) for dir in os.listdir(self.__path) if
                               os.path.isdir(os.path.join(self.__path, dir))]

    # compulsary & run seguentially just to create a job
    def batchGenerator(self):
        itemIndex = random.sample(range(0, len(self.trainDirectory)), self.batchSize)
        result = []
        for i in itemIndex:
            result.append(self.trainDirectory[i])
        return result

    # compulsary & run in parallell among multiple Processor, Each processor execute a 1 job
    def batchProcessor(self, paths):
        X = []
        Y = []
        for path in paths:
            temp = []
            path = os.path.join(path, 'cat_')
            for i in range(5):
                temp.append(ImageOps.readImage(path + str(i) + '.jpg') / 255)
            Y.append(ImageOps.readImage(path + 'result.jpg') / 255)
            X.append(np.array(temp))
        return np.array(X), np.array(Y)


class CocoGenerator:
    def __init__(self, path, batchSize=64, imageSize=64):
        self.__path = path
        self.batchSize = batchSize
        self.imageSize = imageSize

    def init(self):
        trainFilePath = os.path.join(os.path.join(self.__path, 'imageLists'), 'train.txt')
        self.imagesDir = os.path.join(self.__path, 'images')
        self.annotationsDir = os.path.join(self.__path, 'annotations')
        trainFileReader = open(trainFilePath, 'r')
        self.trainFilesList = [line.replace('\n', '') for line in trainFileReader.readlines()]

    # compulsary & run seguentially just to create a job
    def batchGenerator(self):
        itemIndex = random.sample(range(0, len(self.trainFilesList)), self.batchSize)
        result = []
        for i in itemIndex:
            result.append(self.trainFilesList[i])
        return result

    # compulsary & run in parallell among multiple Processor, Each processor execute a 1 job
    def batchProcessor(self, filenames):
        X = []
        Y = []
        for filename in filenames:
            img = ImageOps.readImage(os.path.join(self.imagesDir, filename + '.jpg'))
            img = ImageOps.resizeImage(img, self.imageSize, self.imageSize)
            if len(img.shape) == 2:
                img = np.pad(img.reshape((self.imageSize, self.imageSize, 1)), [(0, 0), (0, 0), (0, 2)],
                             mode='constant', constant_values=0)
            X.append(img)
            Y.append(h5py.File(os.path.join(self.annotationsDir, filename + '.mat'), 'r'))
        return np.array(X), np.array(Y)


class FlowerGenerator:
    def __init__(self, path, batchSize=64):
        self.__path = path
        self.batchSize = batchSize
        self.__trainClasses = {}
        self.isTrain = True
        self.__testClasses = {}
        self.__lastPointer = 0
        self.__imageDir = "jpg"
        self.imageSize = 64

    def init(self):
        fr = open(os.path.join(self.__path, 'files.txt'), 'r')
        fr = fr.readlines()

        for i in range(17):
            self.__trainClasses[str(i)] = []
            self.__testClasses[str(i)] = []
            for line in fr[80 * i:80 * i + 60]:
                self.__trainClasses[str(i)].append(line.replace("\n", ""))

            for line in fr[80 * i + 60:80 * i + 80]:
                self.__testClasses[str(i)].append(line.replace("\n", ""))

    # compulsary & run seguentially just to create a job
    def batchGenerator(self):
        classCount = len(self.__trainClasses.keys())
        resultX = []
        resultY = []
        i = 0
        classIndex = self.__lastPointer % classCount
        while i < self.batchSize:
            itemIndex = random.sample(range(0, len(self.__trainClasses[str(classIndex)]), ), 1)
            resultX.append(self.__trainClasses[str(classIndex)][itemIndex[0]])
            resultY.append(classIndex)
            classIndex = (classIndex + 1) % classCount
            i = i + 1
        self.__lastPointer = classIndex
        return resultX, resultY

    # compulsary & run in parallell among multiple Processor, Each processor execute a 1 job
    def batchProcessor(self, pathsX, Y):
        X = []
        for filename in pathsX:
            img = ImageOps.readImage(os.path.join(os.path.join(self.__path, self.__imageDir), filename))
            img = ImageOps.resizeImage(img, self.imageSize, self.imageSize)
            img = img / 255
            X.append(img)
        return np.array(X), np.array(Y)

    def getTestData(self):
        testX = []
        testY = []
        for key in self.__testClasses.keys():
            tempY = len(self.__testClasses[key]) * [key]
            tempX, tempY = self.batchProcessor(self.__testClasses[key], tempY)
            testX.append(tempX)
            testY.append(tempY)
        testX = np.concatenate(testX)
        testY = np.concatenate(testY)
        return {'x': testX, 'y': testY}
