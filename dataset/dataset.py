import os
import cv2
import random as rd

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class CustomDataset(Sequence):
    def __init__(self,
                 images,
                 labels,
                 batch_size = 32,
                 resize = False,
                 image_size = (64, 64),
                 transform=None,
                 preprocess = False,
                 return_image_path = False):

        self._images = images
        self._labels = labels
        self._batch_size = batch_size
        self._resize = resize
        self._image_size = image_size
        self._transform = transform
        self._preprocess = preprocess
        self._return_image_path = return_image_path
        
        assert len(self._images) == len(self._labels), "Number of images and labels are not same."
        
        
    def __len__(self):
        return len(self._images) // self._batch_size

    def on_epoch_end(self):
        data_to_shuffle = list(zip(self._images, self._labels))
        rd.shuffle(data_to_shuffle)
        self._images, self._labels = zip(*data_to_shuffle)

    def __preprocess__(self, image):
        ## preprocess code goes here
        return image

    def __get_image(self, image_path):
        #image_filepath = self._images[idx]
        image = cv2.imread(image_path) #tf.keras.preprocessing.image.load_img(image_path)
        #image = tf.keras.preprocessing.image.img_to_array(image)
        if self._resize:
            image = cv2.resize(image, self._image_size, interpolation = cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._preprocess:
            image = self.__preprocess__(image)

        if self._transform is not None:
            image = self._transform(image=image)["image"]

        image = image / 255.

        return image

    def __get_label(self, label):
        if label == 'smoke':
            return 1
        else:
            return 0


    def __getitem__(self, idx):

        image_batch = self._images[idx * self._batch_size:(idx + 1) * self._batch_size]
        label_batch = self._labels[idx * self._batch_size:(idx + 1) * self._batch_size]

        images = np.asarray([self.__get_image(img) for img in image_batch])
        labels = np.asarray([self.__get_label(lbl) for lbl in label_batch])

        return images, labels


if __name__ == '__main__':

    from utils import _get_image_labels
    images, labels = _get_image_labels(r'/home/workstaion/workspace/DATASET_ALL/segmented_smoke')
    print(len(images), len(labels))
    #print(images[25], labels[25])
    gen = CustomDataset(images, labels, resize=True)

    for i in gen:
        print(i[0].shape, i[1].shape)
        break