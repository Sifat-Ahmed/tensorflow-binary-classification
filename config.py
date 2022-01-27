import os
import albumentations as A

class Config:

    def __init__(self) -> None:

        self.num_classes = 1
        self.epochs = 5
        self.batch_size = 124
        self.dataset_dir = r'/home/workstaion/Downloads/data/202106_LSIL_NILM_trial1/train'
        self.validation_dir  = r'/home/workstaion/Downloads/data/202106_LSIL_NILM_trial1/validation'


        self.resize = True
        self.image_size = (32, 32)

        self.model_path = r'saved/'+ 'cnn' +'_'+str(self.image_size[0])+'x'+str(self.image_size[1])+'.h5'
        self.learning_rate = 0.001

        self.classification_threshold = 0.8

        self.train_transform = A.Compose(
            [
                A.CLAHE(p=0.2),
                A.RandomGridShuffle(grid=(2, 2), always_apply=False, p=0.1),
                # A.Cutout(num_holes=2, max_h_size=5, max_w_size=5, fill_value=[
                #     0, 0, 0], always_apply=False, p=0.8),
                A.RandomBrightnessContrast(p=0.1),
                A.Blur(p=0.2),
                A.GaussNoise(p=0.1),
                A.RandomGamma(p=0.1),
            ]
        )

        self.val_transform = A.Compose(
            [
                # A.RandomGridShuffle(grid=(2, 2), always_apply=False, p=0.5),
                # A.Cutout(num_holes=2, max_h_size=5, max_w_size=5, fill_value=[
                #     0, 0, 0], always_apply=False, p=0.8),
                # A.RandomBrightnessContrast(p=0.5),
            ]
        )

        self.test_transform = A.Compose(
            [
                # A.Normalize(mean=(0.485, 0.456, 0.406),
                #             std=(0.229, 0.224, 0.225)),
                # #ToTensorV2(),
            ]
        )
