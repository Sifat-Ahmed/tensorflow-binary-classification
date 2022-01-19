import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.callbacks import ModelCheckpoint
from dataset.utils import get_train_test
from dataset.dataset import CustomDataset
from visualize import plot
from config import Config
from collections import defaultdict
from model.model import CNN
from tqdm.keras import TqdmCallback
from tensorflow.keras.models import load_model

def main():

    ## creating the cofig reference to access global parameters
    cfg = Config()

    print('Creating training, validation set')
    ## creating the training and validation dataset
    ## to get the test dataset, set 'return_test=True' and put 'test_size=some_value < 1'
    (train_images, train_labels), (val_images, val_labels) = get_train_test(cfg.dataset_dir,
                                                                            validation_size=0.3,
                                                                            return_test=False)

    ## creating a callback object for the model
    ## saves the model based on the best validation accuracy
    ## Saves the whole model
    ## Model name is in the cfg file
    ## for accuracy the mode is 'max', in case of loss it should be 'min'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=cfg.model_path,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)


    ## creating the dataset generator object for training
    ## it takes a list image paths and labels
    train_dataset = CustomDataset(
        train_images,
        train_labels,
        resize=cfg.resize,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        transform=cfg.train_transform)

    ## creating the validation dataset generator
    val_dataset = CustomDataset(
        val_images,
        val_labels,
        resize=cfg.resize,
        image_size=cfg.image_size,
        transform=cfg.val_transform)

    ## if the saved model already exists then load the model for training
    """
    TODO: compare the model architecture. There is a possibility that you might end up loading a
    different model rather than the model you want because we are loading the whole model rather the 
    weights of the model. So class definition might change. 
    """
    if os.path.isfile(cfg.model_path):
        print('Saved model found and loading')
        model = load_model(os.path.join(os.getcwd(), cfg.model_path))
    else:
        model = CNN(input_shape=(cfg.image_size[0], cfg.image_size[1], 3),
                    num_classes=cfg.num_classes).get_model()

    print('Starting to train...')
    history = model.fit(train_dataset,
                        validation_data= val_dataset,
                        epochs=cfg.epochs,
                        verbose=0,
                        ## there are two callbacks, one for model and another for TQDM. TQDM will only work when the verbose = 0
                        callbacks=[model_checkpoint_callback, TqdmCallback()])

    ## plotting the history and saves the figure in plots folder
    plot(history)
