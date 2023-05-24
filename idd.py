import os
import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

import torch
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

###########################################################################
# CODE COPIED FROM https://github.com/prajjwal1/autonomous-object-detection
###########################################################################


class IDD(torch.utils.data.Dataset):
    def __init__(self, images_path: Path, split: str, transforms=None):
        super(IDD, self).__init__()
        self.root_path = images_path.parent
        assert split in ['train', 'val', 'test'], f'Wrong split selected -> {split}'
        self.split = split
        self.image_paths, self.anno = self.load_image_paths()
        self.transforms = transforms
        self.classes = {
            "person": 0,
            "rider": 1,
            "car": 2,
            "truck": 3,
            "bus": 4,
            "motorcycle": 5,
            "bicycle": 6,
            "autorickshaw": 7,
            "animal": 8,
            "traffic light": 9,
            "traffic sign": 10,
            "vehicle fallback": 11,
            "caravan": 12,
            "trailer": 13,
            "train": 14,
        }
        for k in self.classes.keys():  # Add 1 to let 0 be the background class
            self.classes[k] = self.classes[k] + 1

    def __len__(self):
        return len(self.image_paths)

    def load_image_paths(self):

        print("Loading image and annotation paths of Indian Driving dataset")
        # with open("datalists/idd_images_path_list.txt", "rb") as fp:
        #     idd_image_path_list = pickle.load(fp)
        # with open("datalists/idd_anno_path_list.txt", "rb") as fp:
        #     idd_anno_path_list = pickle.load(fp)

        with open(self.root_path / f"{self.split}.txt") as f:
            img_paths = f.readlines()
        for i in range(len(img_paths)):
            img_paths[i] = img_paths[i].strip("\n")
            img_paths[i] = img_paths[i] + ".jpg"
            img_paths[i] = str(self.root_path / "JPEGImages" / img_paths[i])

        anno_paths = []
        for i in range(len(img_paths)):
            anno_paths.append(img_paths[i].replace("JPEGImages", "Annotations"))
            anno_paths[i] = anno_paths[i].replace(".jpg", ".xml")

        img_paths, anno_paths = sorted(img_paths), sorted(anno_paths)

        return img_paths, anno_paths

    # def get_height_and_width(self, idx):
    #     img_path = os.path.join(img_path, self.image_paths[idx])
    #     img = Image.open(img_path).convert("RGB")
    #     dim_tensor = torchvision.transforms.ToTensor()(img).shape
    #     height, width = dim_tensor[1], dim_tensor[2]
    #     return height, width

    def get_label_bboxes(self, xml_obj):
        xml_obj = ET.parse(xml_obj)
        objects, bboxes = [], []

        for node in xml_obj.getroot().iter("object"):
            object_present = node.find("name").text
            xmin = int(node.find("bndbox/xmin").text)
            xmax = int(node.find("bndbox/xmax").text)
            ymin = int(node.find("bndbox/ymin").text)
            ymax = int(node.find("bndbox/ymax").text)
            objects.append(self.classes[object_present])
            bboxes.append((xmin, ymin, xmax, ymax))
        return Tensor(objects), Tensor(bboxes)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        labels, bboxes = self.get_label_bboxes(self.anno[idx])

        labels = labels.type(torch.int64)
        img_id = Tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        iscrowd = torch.zeros(len(bboxes,), dtype=torch.int64)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target