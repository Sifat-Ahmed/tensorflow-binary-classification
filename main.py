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
    cfg = Config()
    
    history = defaultdict()
    history['val_loss'], history['train_loss'] = list(), list()
    history['val_acc'], history['train_acc'] = list(), list()

    print('Creating training, validation set')
    (train_images, train_labels), (val_images, val_labels) = get_train_test(cfg.dataset_dir, validation_size=0.3, return_test=False)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=cfg.model_path,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)


    train_dataset = CustomDataset(
        train_images,
        train_labels,
        resize=cfg.resize,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        transform=cfg.train_transform)

    val_dataset = CustomDataset(
        val_images,
        val_labels,
        resize=cfg.resize,
        image_size=cfg.image_size,
        transform=cfg.val_transform)


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
                        callbacks=[model_checkpoint_callback, TqdmCallback()])

    plot(history)
    

if __name__ == '__main__':
    main()
