import torch
import torchvision
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
#from datasets import CocoDetection
from tqdm import tqdm


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    print('Inside remove images with no annotations')
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    def _has_bbox_with_no_area(anno):
        return any(obj["area"] <= 1 for obj in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        if _has_bbox_with_no_area(anno):
            print('Object with no area found')
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        print('Reached bottom of _has_valid_annotation()')
        return False

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # This way we transform [x, y, width, height] to [x_min, y_min, x_max, y_max]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # --------------------------------
        # Deactivated as we don't use them
        # and masks set to None
        # -------------------------------
        # segmentations = [obj["segmentation"] if obj["segmentation"] != [] else None for obj in anno]
        # masks = convert_coco_poly_to_mask(segmentations, h, w)
        masks = None
        # ----------------------

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if masks is not None:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if masks is not None:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # TODO: Watch carefully this
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in tqdm(range(len(ds)), desc='Converting dataset to coco api'):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, CocoDetection):  # Custom CocoDetection class
            return dataset.coco
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, known_classes=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.known_classes = known_classes
        self.original_ids_to_new_ids = None
        if known_classes:
            print('-'*50)
            print('Transforming dataset to keep only the categories specified in the config file:')
            self.transform_dataset_class_to_keep_only_known_categories()
            print(self.coco.cats)
            print('-' * 50)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)

        # # If known classes is present, we must do removal of annotations at runtime
        # if self.known_cls_ids:
        #     target['annotations'] = [obj for obj in target['annotations'] if obj["category_id"] in self.known_cls_ids]

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def transform_dataset_class_to_keep_only_known_categories(self):
        categories_orig_id = []
        for cat in self.known_classes:
            categories_orig_id.append({
                'id': cat['orig_id'], 'name': cat['name']
            })

        # Add the dict to translate original IDs to new IDs
        self.original_ids_to_new_ids = {}
        for cls in self.known_classes:
            self.original_ids_to_new_ids[cls['orig_id']] = cls['id']

        # List with only the original id of the categories
        known_classes_original_ids = [x['id'] for x in categories_orig_id]

        # Modify annotations
        new_ann = []
        for ann in self.coco.dataset['annotations']:
            if ann['category_id'] in known_classes_original_ids:
                ann['category_id'] = self.original_ids_to_new_ids[ann['category_id']]
                new_ann.append(ann)

        self.coco.dataset['annotations'] = new_ann

        # Modify images dict
        new_img_ids = set()
        for ann in self.coco.dataset['annotations']:
            new_img_ids.add(ann['image_id'])
        new_images = []
        for img_info in self.coco.dataset['images']:
            if img_info['id'] in new_img_ids:
                new_images.append(img_info)
        self.coco.dataset['images'] = new_images

        # Modify categories
        self.coco.dataset['categories'] = self.known_classes[1:]

        # Create Index with new annotations, images and categories and modify the ids
        self.coco.createIndex()
        self.ids = list(sorted(self.coco.imgs.keys()))


def keep_only_known_classes(coco_obj, known_classes):
    print('Changing the number of classes of coco and recreating index')
    from copy import deepcopy
    copied_coco = deepcopy(coco_obj)
    categories_orig_id = []
    for cat in known_classes:
        categories_orig_id.append({
            'id': cat['orig_id'], 'name': cat['name']
        })

    # Dict to translate original IDs to new IDs
    original_ids_to_new_ids = {}
    for cls in known_classes:
        original_ids_to_new_ids[cls['orig_id']] = cls['id']

    # List with only the original id of the categories
    known_classes_original_ids = [x['id'] for x in categories_orig_id]

    # Modify annotations
    new_ann = []
    for ann in copied_coco.dataset['annotations']:
        if ann['category_id'] in known_classes_original_ids:
            ann['category_id'] = original_ids_to_new_ids[ann['category_id']]
            new_ann.append(ann)

    copied_coco.dataset['annotations'] = new_ann

    # Modify images dict
    new_img_ids = set()
    for ann in copied_coco.dataset['annotations']:
        new_img_ids.add(ann['image_id'])
    new_images = []
    for img_info in copied_coco.dataset['images']:
        if img_info['id'] in new_img_ids:
            new_images.append(img_info)
    copied_coco.dataset['images'] = new_images

    # Modify categories
    copied_coco.dataset['categories'] = known_classes[1:]

    return copied_coco