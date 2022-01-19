## This repository contains the source code of binary classification project

### How to Run

#### 1. Refer to **config.py** for model parameters and configuration

```
num_classes = 1                     ## Number of classes, as this project is a binary classification task
                                    ## By default the value has been set to 1
                                    ## For multiclass, change the value according to number of classes
                                    ## Also, change the criterion accordingly in 'main.py' line no. 90 
```
```
epochs = 50                         ## Number of epochs 
```
```
batch_size = 256                    ## Size of minibatch
```
```
dataset_dir = 'path/to/dataset'  ## put the root directory of the dataset here.
                                    ## This directory should contain two subfolders. 
                                    ## One for smoke images and the other for non smoke images.
```
```
resize = True                       ## By deafult the images will be resized. If it's false, 
                                    ## then the model will be trained on default size of the images
                                    ## and the dataset should not contain images of inconsistent size
```

```
image_size = (32, 32)             ## Size of the target image after resizing,
                                    ## No color channel info is required.
```

```

model_path = "path/to/model"        ## the model is saved according to the name and image size
                                    ## In case of some thing else, update it accordingly
                                    ## by default it saves the model in 'saved' folder.

```
```
learning_rate = 0.001               ## 0.001 is the default value of learning rate. Till now,
                                    ## This seems to be the optimal value. Higher values causes fluctuation
                                    ## and lower values causes slower convergence
                                    ## Anything between 0.001~0.005 is good enough
```
```

classification_threshold = 0.75     ## The output of sigmoid function is either
                                    ## <0.1 or >0.9 so the threshold value can be
                                    ## chosen anything between 0.4~0.8.
                                    ## But choosing a higher value reduces false positives 
                                    ## Which can be seen in exceptional cases
```

There are three different transform functions for training, validation and testing. Training process requires heavy image augmentation. Otherwise, there is a tendency of the model to overfit and learn nothing. For the augmentation purpose, albumentations library has been used.
The functions of the training transform are.

```
- CLAHE                             ## Contrast Limited Adaptive Histogram Equalization
                                    ## Used to balance any distortion in the lighting
- Cutout                            ## Creates small squares in the image and fills them with greyish color
                                    ## Significantly increases the accuracy as the model has 
                                    ## a tendecy to identify white garbage as smoke
- Flip                              ## Flips the images randomly either horizontally or vertically
- RGBShift                          ## Randomly shift the value of RGB layers
- RandomFog                         ## Adds a fogginess in the images
                                    ## Improves the generalization of the model
- RandomBrightnessContrast          ## Randomly changes the brightness/contrast of the images

```
All of the transform functions have two things in common. 

#### 2. Train
Once the config.py has been updated, the model can be trained by running 
```
python main.py                      ## No extra argument is required

or,

python3 main.py 
```
given that you are already in the project folder.

![Training-Validation Accuracy and Loss](plots/figure.png "Accuracy/Loss Curves")

<center> Fig: Training and Validation Curves </center>

