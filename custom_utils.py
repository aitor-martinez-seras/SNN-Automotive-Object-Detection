from typing import List, Dict
from pathlib import Path
import yaml
from typing import Optional
import os
from datetime import datetime

import torch
from torchvision.ops.boxes import box_convert
from torchvision.utils import draw_bounding_boxes
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

import matplotlib.pyplot as plt

plt.style.use('ggplot')


def _load_yaml(path: str or Path) -> dict:
    with open(path) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
    return args


def load_config_dict(dataset_name: str) -> dict:
    # Create paths
    wrk_dir = Path('.')
    conf_dir = wrk_dir / 'configs'

    # Initialize list with available datasets by listing the directory files. Name of the .yaml file must be the same
    # as the string of the dataset name introduced
    available_datasets = [x for x in os.listdir(conf_dir)]
    conf = {}
    # Check for coincidences between the dataset name introduced and the available ones.
    for name in available_datasets:
        if name == str(dataset_name + '.yaml'):
            conf_path = conf_dir / name
            conf = _load_yaml(conf_path)
            break

    # If len == 0, the dataset has not been found so raise error. Else return the configuration.
    if len(conf) == 0:
        raise NameError(f'The name -- {dataset_name} -- is an incorrect dataset name. Only this names are available: '
                        f'{[str(name) for name in available_datasets]}. Change config .yaml config names to be equal to'
                        ' one of the names showed.')
    else:
        return conf


# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss,
            epoch, model, optimizer
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'outputs/best_model.pth')


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


class TargetsCoco:
    def __init__(self):
        pass

    def __call__(self, targets):
        try:
            targets_dict_of_list = {key: [i[key] for i in targets] for key in targets[0]}
        except IndexError:
            # In case no objects are present in the image, we pass a empty list
            # that will be handled in the dataset class
            return []
        # No Segmentation nor id is not used
        targets_dict_of_list.pop("segmentation")
        targets_dict_of_list.pop("id")
        # Convert to tensor and pop the old keys
        targets_dict_of_list["boxes"] = box_convert(torch.as_tensor(targets_dict_of_list["bbox"], dtype=torch.float32),
                                                    in_fmt='xywh',
                                                    out_fmt='xyxy')
        targets_dict_of_list.pop("bbox")
        targets_dict_of_list["labels"] = torch.as_tensor(targets_dict_of_list["category_id"], dtype=torch.int64)
        targets_dict_of_list.pop("category_id")
        targets_dict_of_list["image_id"] = torch.as_tensor(targets_dict_of_list["image_id"], dtype=torch.int64)
        targets_dict_of_list["area"] = torch.as_tensor(targets_dict_of_list["area"], dtype=torch.float32)
        targets_dict_of_list["iscrowd"] = torch.as_tensor(targets_dict_of_list["iscrowd"], dtype=torch.int64)
        return targets_dict_of_list


def save_checkpoint(output_dir, epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'{output_dir}/last_model.pth')


def check_folder_exist(dir_path_str):
    working_dir_path = Path('.')
    dir_path_str = working_dir_path / dir_path_str
    dir_path_str.mkdir(parents=False, exist_ok=True)


def _imgs_have_all_annotations_checker(set_option):
    """
    This method prints the set with the image_ids in the annotations file that are present in the "images"
    but not in the "annotations".
    """
    import json
    CONFIG = load_config_dict('coco')
    if set_option == 'validation':
        path_to_ann_file = CONFIG["ANN_FILE_VAL"]
    elif set_option == 'train':
        path_to_ann_file = CONFIG["ANN_FILE_TRAIN"]
    else:
        raise Exception()

    with open(path_to_ann_file) as f:
        ann_file = json.load(f)
    # Annotations
    set_ids_in_annotations = set()
    for element in ann_file["annotations"]:
        set_ids_in_annotations.add(element["image_id"])
    # Images
    set_ids_in_images = set()
    for element in ann_file["images"]:
        set_ids_in_images.add(element["id"])
    diff_set = set_ids_in_images.difference(set_ids_in_annotations)
    print(diff_set)


def create_img_with_bboxes(image: torch.Tensor, output_pred: Optional[dict], labels_list: List[Dict],
                           scores_list: List[int] = None) -> torch.Tensor:
    # Transform the image to the required type and to the value pixels
    image = (image * 255).to(dtype=torch.uint8)

    if scores_list:  # We have received confidence scores
        fg_boxes_idx = torch.where(output_pred['labels'] != 0)[0]
        list_pred_classes = [f'{labels_list[lab.item()]["name"]} - {scores_list[idx]:.3f}' for idx, lab in enumerate(output_pred['labels'][fg_boxes_idx])]
        return draw_bounding_boxes(image, boxes=output_pred['boxes'][fg_boxes_idx], labels=list_pred_classes)

    # OPTION 1: we have received only proposals or detections
    if isinstance(output_pred, torch.Tensor):
        return draw_bounding_boxes(image, boxes=output_pred)

    # List containing a string for each bbox, representing either the label or the objectness of the bbox
    list_pred_classes = []

    # OPTION 2: we have received proposals and its scores
    if isinstance(output_pred, tuple):
        for score in output_pred[1]:
            list_pred_classes.append(f'{score:.1f}%')
        return draw_bounding_boxes(image, boxes=output_pred[0], labels=list_pred_classes)

    # OPTION 3: we have received either proposals or images from the dataloader
    if not "scores" in output_pred.keys():

        # Images from dataloader
        if "labels" in output_pred.keys():
            list_pred_classes = [f'{labels_list[lab.item()]["name"]}' for lab in output_pred['labels']]
        # Images from dataloader
        else:
            return draw_bounding_boxes(image, boxes=output_pred['boxes'])

    # OPTION 4: we have received detections or images from dataloader, its predicted labels and scores (softmax values)
    else:
        if labels_list is None:
            if "scores" in output_pred.keys():
                for i in range(len(output_pred["labels"])):
                    list_pred_classes.append(f'{output_pred["labels"][i]} {output_pred["scores"][i] * 100:.1f}%')
            else:
                list_pred_classes = [f'{lab.item()}' for lab in output_pred['labels']]

        else:
            if "scores" in output_pred.keys():
                for i in range(len(output_pred["labels"])):
                    list_pred_classes.append(
                        f'{labels_list[output_pred["labels"][i]]["name"]} {output_pred["scores"][i] * 100:.1f}%')
            else:
                list_pred_classes = [f'{labels_list[lab.item()]["name"]}' for lab in output_pred['labels']]

    # This pytorch function requires the image to be in the range [0, 255] as uint8 values. The bboxes must be
    # [xmin, ymin, xmax, ymax] and labels must be a list of strings containing the labels of each bbox.
    # https://pytorch.org/vision/main/generated/torchvision.utils.draw_bounding_boxes.html
    return draw_bounding_boxes(image, boxes=output_pred['boxes'], labels=list_pred_classes)


def check_freezed_modules(model: GeneralizedRCNN):
    _check_freezed_modules(model, to_file=None)
    hyperparams_path = Path('outputs/coco/hyperparams.txt')
    with open(hyperparams_path, "w") as text_file:
        _check_freezed_modules(model, to_file=text_file)


def _check_freezed_modules(model: GeneralizedRCNN, to_file=None):
    total_cnn = 0
    total_fpn = 0
    freezed_cnn = 0
    freezed_fpn = 0
    for param in model.backbone.named_parameters():
        if param[0].startswith('fpn'):
            total_fpn += 1
            if param[1].requires_grad is False:
                freezed_fpn += 1
        elif param[0].startswith('body'):
            total_cnn += 1
            if param[1].requires_grad is False:
                freezed_cnn += 1
        else:
            raise NameError("Add your model's architecture specificities to check_freezed_modules function")

    print(' -------  Analizing freezed modules  -------', file=to_file)
    percentage_freezed_cnn = (freezed_cnn / total_cnn) * 100
    p = f'{percentage_freezed_cnn:02.0f}'
    print(' ---------------------------', file=to_file)
    print(f'| {p}% Backbone is freezed  |', file=to_file)
    print(' ---------------------------', file=to_file)

    percentage_freezed_fpn = (freezed_fpn / total_fpn) * 100
    p = f'{percentage_freezed_fpn:02.0f}'
    print(' ---------------------------', file=to_file)
    print(f'|   {p}% FPN is freezed     |', file=to_file)
    print(' ---------------------------', file=to_file)

    for param in model.rpn.parameters():
        if param.requires_grad is False:
            print(' -------------------------', file=to_file)
            print('|      RPN is freezed     |', file=to_file)
            print(' -------------------------', file=to_file)
            break
    for param in model.roi_heads.parameters():
        if param.requires_grad is False:
            print(' -------------------------', file=to_file)
            print('|   Detector is freezed   |', file=to_file)
            print(' -------------------------', file=to_file)
            break


def print_hyperparameters(model, opt, lr_sched, args):
    _print_hyperparameters(model, opt, lr_sched, args, to_file=None)
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    save_name = ''
    if args.save_name:
        save_name = "_" + args.save_name
    if args.detector_snn or args.rpn_snn:
        hyperparams_path = Path(f'outputs/{args.dataset}/{timestamp}_SNN_hyperparams{save_name}.txt')
    else:
        hyperparams_path = Path(f'outputs/{args.dataset}/{timestamp}_NoSNN_hyperparams{save_name}.txt')
    with open(hyperparams_path, "w") as text_file:
        _print_hyperparameters(model, opt, lr_sched, args, to_file=text_file)


def _print_hyperparameters(model: GeneralizedRCNN, opt, lr_sched, args, to_file=None):
    print('------------------------------------ Hyperparameters ------------------------------------', file=to_file)
    print(f'Batch size: {args.batch_size}', file=to_file)
    print(f'OPTIMIZER:', file=to_file)
    print(opt, file=to_file)
    print(f'SCHEDULER:', file=to_file)
    if isinstance(lr_sched, list):
        for sch in lr_sched:
            print(sch.state_dict(), file=to_file)
    else:
        print(f'    Base LR: \t{lr_sched.base_lrs}', file=to_file)
        print(f'    Gamma: \t{lr_sched.gamma}', file=to_file)
        try:
            print(f'    Step size: \t{lr_sched.step_size}', file=to_file)
        except AttributeError:
            print(f'    Milestones: \t{lr_sched.milestones}', file=to_file)
    print(f'MODEL:', file=to_file)
    print(f'    RPN:', file=to_file)
    print(model.rpn, file=to_file)
    try:
        print(f'    Number of timesteps: {model.rpn.head.num_steps}', file=to_file)
        print(f'    Encoder threshold: {model.rpn.head.p_enc.v_th}', file=to_file)
    except AttributeError:  # In case there is no SNN
        print(f'    No RPN SNN', file=to_file)
    print(f'    Detector:', file=to_file)
    print(model.roi_heads, file=to_file)
    try:
        print(f'    Number of timesteps: {model.roi_heads.box_head_and_predictor.num_steps}', file=to_file)
        print(f'    Encoder threshold: {model.roi_heads.box_head_and_predictor.p_enc.v_th}', file=to_file)
    except AttributeError:  # In case there is no SNN
        print(f'    No Detector SNN', file=to_file)

    print(f'    GeneralizedRCNNTransform:', file=to_file)
    print(model.transform, file=to_file)
    print('-----------------------------------------------------------------------------------------', file=to_file)
    print('------------------------------------------ Args -----------------------------------------', file=to_file)
    print(f'{args}', file=to_file)
    print('-----------------------------------------------------------------------------------------', file=to_file)
