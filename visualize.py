import os

import matplotlib.pyplot as plt

def plot(history, name=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Accuracy and Loss')
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.legend(['train', 'validation'], loc='upper left')

    ax1.set(xlabel='epoch', ylabel='accuracy')
    ax2.set(xlabel='epoch', ylabel='loss')
    ax1.label_outer()
    ax2.label_outer()

    plt.show()
    if name:
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/'+name)