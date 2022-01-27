from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Input
from tensorflow.keras.layers import GlobalMaxPool2D, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras import Model

class Resnet50TL:
    def __init__(self, num_classes,
                 input_shape,
                 weights='imagenet',
                 include_top=False,
                 ):

        if include_top==True and num_classes != 1000 and weights == 'imagenet':
            print('Include_top with imagenet requires num_classes to be 1000, to avoid this use include_top=False')
        
        self._num_classes = num_classes
        self._input_shape = input_shape
        self._weights = weights
        self._include_top = include_top
        self._activation = 'softmax' if self._num_classes > 1 else 'sigmoid'
        self._loss = 'categorical_crossentropy' if self._num_classes > 1 else 'binary_crossentropy'
        self._optimizer = 'adam'
        self._metrics = ['accuracy']

        self._base_model = ResNet50(classes=self._num_classes,
                                    weights=self._weights,
                                    input_shape=self._input_shape,
                                    include_top=self._include_top)

    def _fully_connected(self):
        if self._include_top:
            return self._base_model

        self._inputs = Input(shape = self._input_shape)
        x = self._base_model(self._inputs)
        x = GlobalMaxPool2D()(x)
        x = Dense(256)(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Dense(self._num_classes)(x)
        self._outputs = Activation(self._activation)(x)

        return Model(self._inputs, self._outputs )
        

    def get_model(self):
        self._model = self._fully_connected()
        self._model.compile(optimizer=self._optimizer,
                            loss=self._loss,
                            metrics=self._metrics)
        return self._model


if __name__ == '__main__':
    model = Resnet50TL(num_classes=3, input_shape=(32, 32, 3)).get_model()
    print(model.summary())