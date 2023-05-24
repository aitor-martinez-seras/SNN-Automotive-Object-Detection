from pathlib import Path
import requests
import shutil
from zipfile import ZipFile
from io import BytesIO

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection

import presets
import transforms as T  # CUSTOM TRANSFORM SCRIPT
from custom_utils import collate_fn, load_config_dict
from coco_utils import ConvertCocoPolysToMask, CocoDetection, _coco_remove_images_without_annotations
from idd import IDD


# ------------------------------------------------------------------------------------------
# Dataset loading methods
# ------------------------------------------------------------------------------------------
def download_cityscapes_annotations(config: dict):
    dir_path = Path(config['ANN_DIR'])
    print('Downloading annotations of Cityscapes!')
    url = r'https://tecnalia365-my.sharepoint.com/:u:/g/personal/aitor_martinez_tecnalia_com/EfD21vmwQztJpp_Rg8nB9ecBkKNM3a1uV8ekVeU4TP8OTw?download=1'
    r = requests.get(url)
    zipfile = ZipFile(BytesIO(r.content))
    zipfile.extractall(path=dir_path)
    print('Annotations downloaded')


def check_cityscapes_exist(config):
    img_folder_path = Path(config["IMAGES_DIR"]) / "leftImg8bit"
    if img_folder_path.exists():

        ann_train_path = Path(config["ANN_FILE_TRAIN"])
        ann_val_path = Path(config["ANN_FILE_VAL"])
        if not ann_train_path.exists() or not ann_val_path.exists():
            download_cityscapes_annotations(config)

    else:
        raise ValueError(f'Cityscapes dataset does not exist in the path {config["IMAGES_DIR"]}')


def download_bdd_annotations(config: dict):
    dir_path = Path(config['ANN_DIR'])
    print('Downloading annotations of BDD!')
    url = r'https://tecnalia365-my.sharepoint.com/:u:/g/personal/aitor_martinez_tecnalia_com/EWcPSP09AVVCifrSAd8IPVgB6uTDxhtAkmzXuC87BC2x0w?download=1'
    r = requests.get(url)
    zipfile = ZipFile(BytesIO(r.content))
    zipfile.extractall(path=dir_path)
    print('Annotations downloaded')


def check_bdd_exist(config):
    img_folder_path = Path(config["IMAGES_DIR"])
    if img_folder_path.exists():
        ann_train_path = Path(config["ANN_FILE_TRAIN"])
        ann_val_path = Path(config["ANN_FILE_VAL"])
        if not ann_train_path.exists() or not ann_val_path.exists():
            download_bdd_annotations(config)

    else:
        raise ValueError(f'BDD dataset does not exist in the path {config["IMAGES_DIR"]}')


def create_dataset(dataset_choice: str, set_option: str, transforms, known_classes=None):
    # Load the config
    config = load_config_dict(dataset_choice)

    # Each dataset must be loaded differently
    if dataset_choice == 'cityscapes':

        check_cityscapes_exist(config)

        if known_classes:
            print('Using only known classes from cityscapes')

        t = [ConvertCocoPolysToMask()]
        if transforms is not None:
            t.append(transforms)
        transforms = T.Compose(t)
        if set_option == 'train':
            dataset_class = CocoDetection(
                config['IMAGES_DIR'],
                ann_file=config['ANN_FILE_TRAIN'],
                transforms=transforms,
                known_classes=known_classes
            )
        elif set_option == 'validation':
            dataset_class = CocoDetection(
                config['IMAGES_DIR'],
                ann_file=config['ANN_FILE_VAL'],
                transforms=transforms,
                known_classes=known_classes
            )

        else:
            raise NameError('Choose between "train" and "validation" options.')

    elif dataset_choice == 'bdd':

        t = [ConvertCocoPolysToMask()]
        if transforms is not None:
            t.append(transforms)
        transforms = T.Compose(t)

        if set_option == 'train':
            dataset_class = CocoDetection(
                Path(config['IMAGES_DIR']) / 'train',
                ann_file=config['ANN_FILE_TRAIN'],
                transforms=transforms,
                known_classes=known_classes
            )
        elif set_option == 'validation':
            dataset_class = CocoDetection(
                Path(config['IMAGES_DIR']) / 'val',
                ann_file=config['ANN_FILE_VAL'],
                transforms=transforms,
                known_classes=known_classes
            )

        else:
            raise NameError('Choose between "train" and "validation" options.')

    elif dataset_choice == 'idd':

        if set_option == 'train':
            dataset_class = IDD(
                images_path=Path(config['IMAGES_DIR']),
                split='train',
                transforms=transforms
            )
        elif set_option == 'validation':
            dataset_class = IDD(
                images_path=Path(config['IMAGES_DIR']),
                split='val',
                transforms=transforms
            )

        else:
            raise NameError('Choose between "train" and "validation" options.')

    else:
        raise NameError(f'{dataset_choice} is an incorrect dataset name. '
                        f'Check config .yaml config names to be equal to the introduced name')

    return dataset_class


def create_data_loader(dataset: Dataset, batch_size: int, num_workers=0, shuffle=False,
                       sampler=None):
    if sampler is not None:
        print(f'Shuffle will be set to False as a Sampler is defined: {sampler.__class__}')
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # Accelerates data transfer to GPU
        sampler=sampler
    )
    return loader


def get_img_transform_coco(train: bool, data_augm: str, noise_intensity=0.05):
    """
    Defines the transformations that will be made to the PIL image. If training, also some augmentations are done.
    :param train: bool, if True some augmentations are done to the image
    :return: Compose object
    """
    if train:
        return presets.DetectionPresetTrain(data_augmentation=data_augm)
    else:
        return presets.DetectionPresetEval(data_augmentation=data_augm, noise_intensity=noise_intensity)

if __name__ == '__main__':
    dataset_name = 'cityscapes'
    num_iterations_of_loader = 20

    data_validation = create_dataset(dataset_choice=dataset_name, set_option='validation',
                                     transforms=get_img_transform_coco(train=False, data_augm="hflip"))
    # data_train = create_dataset(dataset_choice=dataset_name, set_option='train',
    #                             transforms=get_img_transform_coco(train=False, data_augm="hflip"))

    print(data_validation)
    dataloader = create_data_loader(data_validation, batch_size=2, num_workers=0, shuffle=True)

    classes = load_config_dict(dataset_name)['CLASSES']

    print(len(data_validation))

    from custom_utils import create_img_with_bboxes
    import matplotlib.pyplot as plt

    br = False
    iterable = iter(dataloader)
    for i in range(num_iterations_of_loader):
        images, targets = next(iterable)
        # for j, img in enumerate(images):
        #     img = create_img_with_bboxes(img, targets[j], classes)
        #     plt.imshow(img.permute(1, 2, 0))
        #     plt.savefig(f'img_{i}_{j}.png')
        #     plt.close()
        # for label in targets[1]['labels']:
        #     if label == 10:
        #         br = True
        #         break
        # if br:
        #     break

    img = create_img_with_bboxes(images[1], targets[1], classes)
    plt.imshow(img.permute(1, 2, 0))
    plt.show()