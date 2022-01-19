from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


class CNN:
    def __init__(self,
                 input_shape,
                 num_classes,
                 loss = 'binary_crossentropy',
                 optimizer = 'adam'
                 ):
        self._model = Sequential()
        weight_decay = 0.0005

        self._model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape,
                         kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())
        self._model.add(Dropout(0.2))

        self._model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())

        self._model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())

        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())
        self._model.add(Dropout(0.2))

        self._model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())

        self._model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())

        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())
        self._model.add(Dropout(0.2))

        self._model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())

        self._model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())

        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())
        self._model.add(Dropout(0.2))

        self._model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())
        self._model.add(Dropout(0.2))

        self._model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())

        self._model.add(MaxPooling2D(pool_size=(2, 2)))

        self._model.add(GlobalMaxPooling2D())

        self._model.add(Dense(256, kernel_regularizer=regularizers.l2(weight_decay)))
        self._model.add(Activation('relu'))
        self._model.add(BatchNormalization())

        self._model.add(Dropout(0.2))
        self._model.add(Dense(num_classes))
        self._model.add(Activation('sigmoid'))

        self._model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def get_model(self, ):
        return self._model
