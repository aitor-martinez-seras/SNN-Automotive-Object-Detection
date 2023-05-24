from pathlib import Path
from copy import deepcopy

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.ops
from torchvision.ops import boxes as box_ops
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from custom_utils import load_config_dict, create_img_with_bboxes
from datasets import create_dataset, get_img_transform_coco, create_data_loader


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch SNN Detection Training", add_help=add_help)

    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset name",
                        choices=['cityscapes', 'bdd'])
    parser.add_argument("-f", "--file", type=str, required=True,
                        help="path to the file with the detections")
    parser.add_argument("--only-known-cls", default=False, action="store_true", dest="only_known_cls",
                        help="Pass this to only use the classes in the .yaml file of the dataset")
    parser.add_argument("-s", "--save-images", default=0, type=int, dest="save_images",
                        help="Number of images to save with boxes of BG with possible objects")
    parser.add_argument("-iou", "--iou-thr", default=0.05, type=float, dest="iou_thr",
                        help="Max IoU between the predictions and new objects for that possible new objects to be"
                             "discarded")
    parser.add_argument("-sc", "--score-thr", default=0.25, type=float, dest="score_thr",
                        help="Score thr for the new obj")
    parser.add_argument("-nms", "--nms-thr", default=0.5, type=float, dest="nms_thr",
                        help="The IoU thr for the NMS in new object discovery")
    parser.add_argument("-max", "--max-detections", default=0, type=int, dest="max_detections",
                        help="Max number of detections per image")
    parser.add_argument("--plot-only-new-objs", default=False, action="store_true", dest="plot_only_new_objs",
                        help="Plot only img of preds")
    parser.add_argument("--plot-all-in-one-img", default=False, action="store_true", dest="plot_all_in_one_img",
                        help="Plot only img of preds")
    return parser


def main(args):
    # -----------------------
    # Configuration
    # -----------------------
    # Load config. Added datasets must have same variables defined
    CONFIG = load_config_dict(args.dataset)
    try:
        output_dir_path = Path(CONFIG["OUT_DIR"])
        NUM_CLASSES = CONFIG["NUM_CLASSES"]
        classes = CONFIG["CLASSES"]
    except KeyError as e:
        raise KeyError(f'Check that the keys in the .yaml files of config are correct: {e}')
    # If we have defined known class, we need to change the known classes var
    known_classes = None
    if args.only_known_cls:
        try:
            known_classes = CONFIG['KNOWN_CLASSES']
            class_names = [x['name'] for x in known_classes]  # Includes BG
        except KeyError:
            pass
    else:
        class_names = [x['name'] for x in classes]  # Includes BG

    # Load detections
    detections_path = Path(args.file)
    detections = torch.load(detections_path)

    path_for_figures = output_dir_path / f'new_objects_{args.dataset}'
    path_for_figures.mkdir(exist_ok=True)

    ##################################################
    # NEW OBJECT DISCOVERY
    ##################################################

    if not args.compute_metrics and not args.save_images:
        raise NotImplementedError('Indicate either metrics or save_images')

    print('########################')
    print('NEW OBJECT DISCOVERY')
    print('########################')

    # Remove bg boxes overlapping more than iou_thr with detections
    iou_thr = args.iou_thr
    print(f'IoU threshold: {iou_thr}')
    for detection in tqdm(detections):

        # Identify BG and non BG boxes
        idx_bg_boxes = np.where(detection['labels'] == 0)[0]
        idx_non_bg_boxes = np.where(detection['labels'] != 0)[0]
        bg_boxes = detection['boxes'][idx_bg_boxes]
        non_bg_boxes = detection['boxes'][idx_non_bg_boxes]

        # Check IoU between BG and non BG
        bg_boxes_iou = torchvision.ops.box_iou(bg_boxes, non_bg_boxes)

        # Define the BG boxes to be removed. Every position of the IoU array is one BG bbox's IoU
        # against the non_bg_boxes. So we iterate per each BG bbox and find wheter it has some non_bg bbox
        # overlapping more than a certain thr. If any bbox is overlapping more then the thr (one_bg_box_ious.any()),
        # then include that bbox in the array of bboxes to be removed
        to_be_removed = []
        for i, one_bg_box_ious in enumerate(bg_boxes_iou):
            # In this if we can set the threshold for the max IoU to have to remove a BG box
            one_bg_box_ious = one_bg_box_ious > iou_thr
            if one_bg_box_ious.any():
                to_be_removed.append(i)

        # Keep only the boxes not in the to_be_removed array
        keep = np.array(list(set(range(len(idx_bg_boxes))).difference(set(to_be_removed))))
        if keep.any():
            idx_bg_boxes_to_keep = idx_bg_boxes[keep]
            idx_to_keep = np.concatenate((idx_non_bg_boxes, idx_bg_boxes_to_keep))
        else:
            idx_to_keep = idx_non_bg_boxes
        for k in detection.keys():
            if k not in ['proposals', 'objectness']:
                detection[k] = detection[k][idx_to_keep]

    # Now create an score for each bbox based on the proposals and the IoU of each BG bbox against this proposals
    #   For the moment, I do it in other loop

    # Cityscapes = 2048 × 1024 (W x H) => ego bounding box = [0.15xW, 0.8xH, W, H]
    # BDD = 1280 x 720 (W x H) => ego bounding box = [0, 0.9xH, W, H]
    if args.dataset == 'cityscapes':
        W, H = 2048, 1024
        ego_bbox = torch.tensor([int(0.15 * W), int(0.8 * H), W, H], dtype=torch.int32).unsqueeze(0)
    elif args.dataset == 'bdd':
        W, H = 1280, 720
        ego_bbox = torch.tensor([0, int(0.9 * H), W, H], dtype=torch.int32).unsqueeze(0)
    else:
        raise NameError

    # Create an score for each bbox based on the proposals and the IoU of each BG bbox against this proposals
    nms_thr = args.nms_thr
    for detection in tqdm(detections):
        # Identify BG and non BG boxes
        idx_bg_boxes = np.where(detection['labels'] == 0)[0]
        idx_non_bg_boxes = np.where(detection['labels'] != 0)[0]
        bg_boxes = detection['boxes'][idx_bg_boxes]
        non_bg_boxes = detection['boxes'][idx_non_bg_boxes]

        one_img_proposals = detection['proposals']
        one_img_objectness = detection['objectness']
        bg_boxes_iou_with_proposals = torchvision.ops.box_iou(bg_boxes, one_img_proposals)

        # New object score
        new_obj_scores_per_proposal = bg_boxes_iou_with_proposals * one_img_objectness

        # Compute the new object scores
        new_object_scores = torch.sum(new_obj_scores_per_proposal, dim=1)

        # Use NMS suppression
        keep_bg = box_ops.batched_nms(bg_boxes, new_object_scores, detection['labels'][idx_bg_boxes], nms_thr)

        # Remove the supressed new objects
        new_object_scores = new_object_scores[keep_bg]
        nms_bg_boxes = bg_boxes[keep_bg]
        all_boxes = torch.cat([non_bg_boxes, nms_bg_boxes])
        all_labels = torch.cat([detection['labels'][idx_non_bg_boxes], detection['labels'][idx_bg_boxes][keep_bg]])
        all_scores = torch.cat([detection['scores'][idx_non_bg_boxes], detection['scores'][idx_bg_boxes][keep_bg]])
        detection['new_object_scores'] = new_object_scores
        detection['boxes'] = all_boxes
        detection['labels'] = all_labels
        detection['scores'] = all_scores

        # Remove objects in the ego car
        # Boxes are [x1, y1, x2, y2] and referenced in absolute pixels to the top-left corner of the image
        # Therefore we must remove boxes which lay in the ego car zone. We define this zone relative to the image
        # height and width
        # ego bounding box = [x_ego1, y_ego1, x_ego2,  y_ego2]
        # If a bounding box overlaps the car, it must be eliminated
        # Cityscapes = 2048 × 1024 (W x H) => ego bounding box = [0.15xW, 0.8xH, W, H]
        # BDD = 1280 x 720 (W x H) => ego bounding box = [0, 0.9xH, W, H]

        # Identify again BG and non BG boxes
        idx_bg_boxes = np.where(detection['labels'] == 0)[0]
        idx_non_bg_boxes = np.where(detection['labels'] != 0)[0]
        bg_boxes = detection['boxes'][idx_bg_boxes]
        non_bg_boxes = detection['boxes'][idx_non_bg_boxes]

        bg_boxes_iou_with_ego_car = torchvision.ops.box_iou(bg_boxes, ego_bbox)
        keep_bg = torch.where(bg_boxes_iou_with_ego_car == 0)[0]

        # Remove the supressed new objects
        new_object_scores = new_object_scores[keep_bg]
        # detection['new_object_scores'] = new_object_scores
        no_ego_car_overlapping_bg_boxes = detection['boxes'][idx_bg_boxes][keep_bg]
        no_ego_car_overlapping_bg_labels = detection['labels'][idx_bg_boxes][keep_bg]
        no_ego_car_overlapping_bg_scores = detection['scores'][idx_bg_boxes][keep_bg]

        # Limit the number of detections to only the best scores
        if args.max_detections:
            # detection['new_object_scores'] = detection['new_object_scores'][:args.max_detections]
            new_object_scores = new_object_scores[:args.max_detections]
            no_ego_car_overlapping_bg_boxes = no_ego_car_overlapping_bg_boxes[:args.max_detections]
            no_ego_car_overlapping_bg_labels = no_ego_car_overlapping_bg_labels[:args.max_detections]
            no_ego_car_overlapping_bg_scores = no_ego_car_overlapping_bg_scores[:args.max_detections]

        all_boxes = torch.cat([non_bg_boxes, no_ego_car_overlapping_bg_boxes])
        all_labels = torch.cat([detection['labels'][idx_non_bg_boxes], no_ego_car_overlapping_bg_labels])
        all_scores = torch.cat([detection['scores'][idx_non_bg_boxes], no_ego_car_overlapping_bg_scores])
        detection['new_object_scores'] = new_object_scores
        detection['boxes'] = all_boxes
        detection['labels'] = all_labels
        detection['scores'] = all_scores

    # PLOT
    # Load the dataset
    dataset = create_dataset(
        args.dataset, set_option='validation',
        transforms=get_img_transform_coco(train=False, data_augm='None'),
        known_classes=known_classes
    )
    dataloader = create_data_loader(dataset, batch_size=1, shuffle=False)

    max_idx = args.save_images
    score_thr = args.score_thr
    print(f'Score threshold: {score_thr}')
    if args.dataset == 'cityscapes':
        fontsize = 20
    else:
        fontsize = 15
    font = '/usr/share/fonts/open-sans/OpenSans-Semibold.ttf'
    color_known = 'green'
    color_unk = 'red'
    for img_idx, (image, target) in enumerate(dataloader):

        image = image[0]
        target = target[0]

        H, W = image.shape[1:]

        # To plot BG bboxes only with the "new object score" metric
        idx_bg_boxes = torch.where(detections[img_idx]['labels'] == 0)[0]
        bg_boxes = detections[img_idx]['boxes'][idx_bg_boxes]

        # To put a threshold in the score
        new_obj_score_one_img = detections[img_idx]['new_object_scores']
        idx_bboxes_above_thr = torch.where(new_obj_score_one_img >= score_thr)[0]
        bg_boxes = bg_boxes[idx_bboxes_above_thr]
        new_obj_score = [f'{n.item():.2f}' for n in new_obj_score_one_img[idx_bboxes_above_thr]]

        # new_obj_score = [f'{s:.2f}' for s in detections[img_idx]['new_object_scores']]
        image2 = (image * 255).to(dtype=torch.uint8)

        if args.plot_only_new_objs:

            img = draw_bounding_boxes(image2, boxes=bg_boxes, labels=new_obj_score, width=2,
                                      font=font, font_size=fontsize)

            plt.imshow(img.permute(1, 2, 0))
            plt.axis('off')
            plt.savefig(path_for_figures / f'{str(target["image_id"][0].item()).zfill(12)}.png', dpi=300,
                        bbox_inches='tight', pad_inches=0)
            plt.close()

        elif args.plot_all_in_one_img:

            bg_labels = ['unk'] * len(new_obj_score)

            # Extract predictions
            idx_fg_boxes = torch.where(detections[img_idx]['labels'] != 0)[0]
            fg_boxes = detections[img_idx]['boxes'][idx_fg_boxes]
            predicted_labels = [class_names[pred] for pred in detections[img_idx]['labels'][idx_fg_boxes]]

            # Fuse preds with new objects
            boxes = torch.cat((fg_boxes, bg_boxes))
            labels = predicted_labels + bg_labels
            colors = [color_known] * len(fg_boxes) + [color_unk] * len(bg_boxes)

            img = draw_bounding_boxes(image2, boxes=boxes, labels=labels, width=2, colors=colors,
                                      font=font, font_size=fontsize)
            plt.imshow(img.permute(1, 2, 0))
            plt.axis('off')
            plt.savefig(path_for_figures / f'{str(target["image_id"][0].item()).zfill(12)}.png', dpi=300,
                        bbox_inches='tight', pad_inches=0)
            plt.close()

        else:
            # new_obj_score = [f'{s:.2f}' for s in detections[img_idx]['new_object_scores']]
            image2 = (image * 255).to(dtype=torch.uint8)
            img = draw_bounding_boxes(image2, boxes=bg_boxes, labels=new_obj_score, width=2,
                                      font=font, font_size=fontsize)
            # figsize = (constant * (Height Width relation) * n_cols, constant * n_rows)
            fig, axes = plt.subplots(
                1, 2, figsize=(6 * (W / H) * 2, 6 * 1), tight_layout=True)

            # To plot BG with the rest of the predictions
            # img = create_img_with_bboxes(image, detections[img_idx], classes)

            # Extract predictions to plot
            idx_fg_boxes = torch.where(detections[img_idx]['labels'] != 0)[0]
            fg_boxes = detections[img_idx]['boxes'][idx_fg_boxes]
            predicted_labels = [class_names[pred] for pred in detections[img_idx]['labels'][idx_fg_boxes]]
            img_preds = draw_bounding_boxes(image2, boxes=fg_boxes, labels=predicted_labels, width=2, font_size=10)

            # To plot targets
            # img = create_img_with_bboxes(image, target, classes)

            axes[0].imshow(img.permute(1, 2, 0))
            axes[0].set_title("Image with new object proposals", y=0.99)
            axes[0].set_axis_off()

            axes[1].imshow(img_preds.permute(1, 2, 0))
            axes[1].set_title("Image with predictions", y=0.99)
            axes[1].set_axis_off()
            fig.savefig(path_for_figures / f'{str(target["image_id"][0].item()).zfill(12)}.png', dpi=300)
            plt.close()

        print(f'{img_idx} images out of {max_idx} processed')
        if img_idx > max_idx:
            print(f'{max_idx} images saved or represented, exiting program!')
            break

    with open(path_for_figures / 'params.txt', mode='w') as f:
        print(f'IoU thr:\t{iou_thr}', file=f)
        print(f'NMS thr:\t{nms_thr}', file=f)
        print(f'Score thr:\t{score_thr}', file=f)


if __name__ == '__main__':
    main(get_args_parser().parse_args())
