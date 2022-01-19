import matplotlib.pyplot as plt

def plot(history, name='figure.png'):
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    # # "Loss"
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

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

    #plt.show()
    plt.savefig('plots/'+name)