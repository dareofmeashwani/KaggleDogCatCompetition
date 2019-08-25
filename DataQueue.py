import os
from multiprocessing import Process, Lock, Semaphore
from multiprocessing import Value, cpu_count
from multiprocessing import Queue
import signal
from inspect import signature



class DataQueue:
    def __init__(self, generator, qsize=16, childCount=6):
        self.__size = qsize
        self.__childCount = childCount
        self.__status = Value('b', False)
        self.__generator = generator
        if (hasattr(self.__generator, 'batchGenerator') and hasattr(self.__generator, 'batchProcessor')) == False:
            raise Exception('batchGenerator and batchProcessor in Generator object is mandatory')

    def loadBatch(self, status, inputSM, batchSM):
        while status.value:
            inputSM['emptySemaphore'].acquire()
            inputSM['mutex'].acquire()
            data = inputSM['q'].get()
            inputSM['mutex'].release()
            inputSM['fullSemaphore'].release()
            sig = signature(self.__generator.batchProcessor)
            if len(sig.parameters) == 1:
                data = self.__generator.batchProcessor(list(data))
            else:
                data = self.__generator.batchProcessor(*data)
            batchSM['fullSemaphore'].acquire()
            batchSM['mutex'].acquire()
            batchSM['q'].put(data)
            batchSM['mutex'].release()
            batchSM['emptySemaphore'].release()

    def getBatch(self):
        #if self.__status.value and self.__batchSM['q'].qsize() != 0 and self.__inputSM['q'].qsize() != 0:
        self.__batchSM['emptySemaphore'].acquire()
        self.__batchSM['mutex'].acquire()
        batch = self.__batchSM['q'].get()
        self.__batchSM['mutex'].release()
        self.__batchSM['fullSemaphore'].release()
        return batch

    def monitor(self, status, inputSM):
        while status.value:
            data = tuple(self.__generator.batchGenerator())
            inputSM['fullSemaphore'].acquire()
            inputSM['mutex'].acquire()
            inputSM['q'].put(data)
            inputSM['mutex'].release()
            inputSM['emptySemaphore'].release()

    def start(self):
        if hasattr(self.__generator, 'init'):
            self.__generator.init()
        self.__status.value = True
        self.__inputSM = {
            "q": Queue(),
            "emptySemaphore": Semaphore(value=0),
            "fullSemaphore": Semaphore(value=self.__size),
            "mutex": Lock()
        }
        self.__batchSM = {
            "q": Queue(),
            "emptySemaphore": Semaphore(value=0),
            "fullSemaphore": Semaphore(value=self.__size),
            "mutex": Lock()
        }

        childCount = self.__childCount if self.__childCount != - 1 else cpu_count()
        self.__childProcess = []
        self.__mProcess = Process(target=self.monitor, args=(self.__status, self.__inputSM))
        self.__mProcess.start()
        for i in range(childCount):
            p = Process(target=self.loadBatch, args=(self.__status, self.__inputSM, self.__batchSM))
            p.start()
            self.__childProcess.append(p)

    def stop(self):
        self.__status.value = False
        for cProcess in self.__childProcess:
            if cProcess.pid is not 0:
                try:
                    os.kill(cProcess.pid, signal.SIGTERM)
                    print("successfully killed child process", cProcess.pid)
                except:
                    print("Access Denied can't kill child process", cProcess.pid)
        try:
            os.kill(self.__mProcess.pid, signal.SIGTERM)
            print("successfully killed monitor process", self.__mProcess.pid)
        except:
            print("Access Denied can't kill monitor process", self.__mProcess.pid)
        print("Job Queue size at stop", self.__inputSM['q'].qsize())
        print("batch Queue size at stop", self.__batchSM['q'].qsize())
        self.__inputSM['q'].close()
        self.__batchSM['q'].close()

    def getCurrentBatchQueueSize(self):
        return self.__batchSM['q'].qsize()

    def isEmpty(self):
        return self.__batchSM['q'].qsize() == 0

    def getCurrentJobQueueSize(self):
        return self.__inputSM['q'].qsize()
