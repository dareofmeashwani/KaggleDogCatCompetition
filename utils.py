import numpy as np
import pickle

def save_model(model,filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    pickle_in = open(filename, 'rb')
    return pickle.load(pickle_in)

def load_encoding_model(filename,encode='ascii'):
    pickle_in = open(filename, 'rb')
    return pickle.load(pickle_in,encoding=encode)

def convert_to_onehot(y_,n_classes):
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]

def back_from_onehot(y):
    x=np.argmax(y, axis=1)
    return x

def sigmoid(x):
    import numpy as np
    return 1 / (1 + np.exp(-x))

def get_confusion_matrix(actual,pred,n_classes,class_names,asText=False):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=actual,  # True class for test-set.
                          y_pred=pred)  # Predicted class.
    if asText==True:
        # Print the confusion matrix as text.
        for i in range(n_classes):
            # Append the class-name to each line.
            class_name = "({}) {}".format(i, class_names[i])
            print(cm[i, :], class_name)
    return cm

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix'):
    import matplotlib.pyplot as plt
    import itertools
    import numpy as np
    #cmap=plt.cm.get_cmap('RdBu')
    cmap = plt.cm.get_cmap('Blues')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_correlation_matrix(correlations,names):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 22, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()
    return

def maybe_download_and_extract(DATA_URL):
    import os, urllib, sys ,tarfile
    dest_directory = 'model'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()


        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

