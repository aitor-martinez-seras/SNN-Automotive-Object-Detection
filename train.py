r"""SNN Object Detection Training.
To run in a multi-gpu environment, use the distributed launcher::
python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env train.py --world-size $NGPU
The default hyperparameters are tuned for training on 2 gpus and 2 images per gpu.
The formula to calculate LR is 0.00125 * batch_size * n_gpus
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --world-size 2
    --lr 0.02 --batch-size 2 --world-size 2
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
"""
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
import torch.utils.data.distributed as distributed
from torch.utils.data import SequentialSampler, RandomSampler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import create_dataset, create_data_loader, get_img_transform_coco
from model import create_model
from custom_utils import (
    check_folder_exist, load_config_dict,
    create_img_with_bboxes, check_freezed_modules, print_hyperparameters
)
from utils import MetricLogger, reduce_dict, init_distributed_mode, save_on_master
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch SNN Detection Training", add_help=add_help)

    parser.add_argument("-d", "--dataset", default="cityscapes", type=str, required=True, help="dataset name",
                        choices=['cityscapes', 'bdd', 'idd', 'pascal'])
    parser.add_argument("-t-rpn", "--rpn-steps", default=12, dest='num_steps_rpn', type=int,
                        help="number of total steps of the RPN")
    parser.add_argument("-t-det", "--det-steps", default=16, dest='num_steps_detector', type=int,
                        help="number of total steps of the detector")
    parser.add_argument("--save-name", default="", type=str, dest='save_name', help="model_{save_name}_{epoch}.pth")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=2, type=int,
                        help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=40, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=1, type=int, metavar="N",
                        help="number of data loading workers (default: 1)")
    parser.add_argument("--opt", default="AdamW", type=str, help="optimizer. Options: AdamW and SGD")
    parser.add_argument("--lr", default=0.0025, type=float,
                        help="initial learning rate, 0.0025 is the default value for training"
                             " on 1 gpu and 2 images_per_gpu")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=0.0001, type=float, metavar="W",
                        help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--lr-decay-milestones", default=[], type=int, nargs='+',
                        dest="lr_decay_milestones", help="lr decay milestones")
    parser.add_argument("--lr-decay-step", default=0, type=int, dest="lr_decay_step", help="lr decay step")
    parser.add_argument("--lr-decay-rate", default=0, type=float, dest="lr_decay_rate", help="lr decay rate")
    parser.add_argument("--constant-lr-scheduler", default=0, type=float, dest="constant_lr_scheduler",
                        help="Use ConstantLR to decrease the LR the first epoch by the factor specified")
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--start-epoch", default=1, type=int, dest="start_epoch", help="start epoch")
    parser.add_argument("--trainable-backbone-layers", dest="trainable_backbone_layers", default=0, type=int,
                        help="number of trainable layers of backbone")
    parser.add_argument("--world-size", default=2, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    # Model loading options
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--load-model", default="", type=str, dest="load_model", help="path of the model to load")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Can only be set if no SNN is used, and in that case the pretrained weights for"
                             "RPN and Detector will be used")
    parser.add_argument("--not-pretrained-fpn", action="store_false", default=True, dest="pretrained_fpn",
                        help="Do not use fpn pretraining from Coco")
    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use torch.cuda.amp for mixed precision training")
    # Testing and image saving options
    parser.add_argument("--validate-every-n-epochs", dest="validate_every_n_epochs", type=int, default=1,
                        help="Peform validation only every N epochs")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true", default=False)
    parser.add_argument("--plot-images", dest="plot_images",action="store_true", default=False,
                        help="Only plot images with predictions, requires weights")
    parser.add_argument("--save-images", dest="save_images", action="store_true", default=False,
                        help="Save images instead of plotting them")
    parser.add_argument("--save-images-option", dest="save_images_option", type=str, default='imgs_and_preds',
                        choices=['imgs_and_preds', 'one_img_preds'], help="Options for image saving")
    # Freezing and SNN using options
    parser.add_argument("--freeze-fpn", default=False, action="store_true", dest="freeze_fpn",
                        help="pass to freeze the Feature Pyramid Network")
    parser.add_argument("--freeze-rpn", default=False, action="store_true", dest="freeze_rpn",
                        help="pass to freeze the RPN")
    parser.add_argument("--freeze-detector", default=False, action="store_true", dest="freeze_detector",
                        help="pass to freeze the detector part")
    parser.add_argument("--rpn-snn", dest="rpn_snn", action="store_true", default=False,
                        help="Implement the RPN as a SNN")
    parser.add_argument("--detector-snn", dest="detector_snn", action="store_true", default=False,
                        help="Implement FasterRCNN detector as SNN")
    # Training enhancing options
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--data-augmentation", dest="data_augmentation",default="hflip", type=str,
                        help="data augmentation policy (default: hflip)")

    # New Object Discovery (NOD)
    parser.add_argument("--only-known-cls", default=False, action="store_true", dest="only_known_cls",
                        help="Pass this to only use the classes in the .yaml file of the dataset")
    parser.add_argument("--only-one-bbox", default=False, action="store_true", dest="only_one_bbox",
                        help="Pass this to only use one bbox prediction in ROIHeads module")
    parser.add_argument("-ext-prop-det", "--extract-proposals-and-detections", nargs="+", default=[],
                        dest="extract_proposals_and_detections",
                        help="Pass train and/or test to extract all information for NOD"
                             "train and/or test data and save to a file")
    parser.add_argument("-n-img", "--max-num-images-for-nod", type=int, default=20000, dest="max_num_images_for_nod",
                        help="Max number of images to extract info for NOD")
    parser.add_argument("--rm-bg", default=False, action="store_true", dest="rm_bg",
                        help="Removes bg predictions from the detections")
    parser.add_argument("--extract-spike-rates",  nargs="+", default=[], dest="extract_spike_rates",
                        help="Pass train and/or test to extract spike rates and FLOPs of "
                             "train and/or test data and save to a file")
    parser.add_argument("--add-noise", type=str, default='', dest="add_noise", help="Types of noises",
                        choices=['gaussian', 'rain','light-rain', 'heavy-rain'])
    parser.add_argument("--noise-intensity", type=float, default=0.05, dest="noise_intensity",
                        help="How much variance for gaussian noise or how many raindrops in case rain is used")

    return parser


def log_training_losses_to_writer(metric_logger: MetricLogger, global_step: int, full_epoch_stats: bool):
    for k, v in metric_logger.meters.items():
        if full_epoch_stats:
            tag = f'Loss/Epochs/{k}'
            value = v.global_avg
        else:
            tag = f'Loss/MiniBacth/{k}'
            value = v.avg
        train_writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)


def log_validation_losses_to_writer(metric_logger: MetricLogger, global_step: int):
    for k, v in metric_logger.meters.items():
        tag = f'Loss/Epochs/{k}'
        value = v.global_avg
        val_writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)


def train_one_epoch(model, device, train_data_loader, optimizer: torch.optim.Optimizer,
                    epoch: int, print_freq: int, scaler: torch.cuda.amp.GradScaler):
    print('Training')
    model.train()
    header = f"Epoch: [{epoch}]"
    metric_logger = MetricLogger(delimiter="  ")

    minibatch_count = 0
    for images, targets in metric_logger.log_every(train_data_loader, print_freq, header):

        # Format images and targets as the model's forward takes them:
        # images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]]. Each position of the list is an image.
        # Each image is a Tensor [ch, H, W]
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # set_to_none=True recommended by pytorch official performance tuning guide
        optimizer.zero_grad(set_to_none=True)

        # Backwards pass
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)  # Distributed

        # Keep track of losses with Tensorboard every print_freq minibatches
        if minibatch_count % print_freq == 0 or minibatch_count == len(train_data_loader) - 1:
            log_training_losses_to_writer(
                metric_logger, global_step=(epoch * len(train_data_loader) + minibatch_count), full_epoch_stats=False
            )

        minibatch_count += 1

    # Log global avg losses for the epoch
    log_training_losses_to_writer(
        metric_logger, global_step=epoch, full_epoch_stats=True
    )

    return metric_logger.meters['loss'].global_avg


def validate_one_epoch(model, device, val_data_loader, epoch: int,
                       print_freq: int, scaler: torch.cuda.amp.GradScaler):
    print('Validating')
    model.train()
    header = f"Epoch: [{epoch}]"
    metric_logger = MetricLogger(delimiter="  ")

    for images, targets in metric_logger.log_every(val_data_loader, print_freq, header):

        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            with torch.no_grad():  # We are validating, we do not use the gradients
                loss_dict = model(images, targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)  # Distributed

        # Don't backpropagate losses, as we are validating

    # Log global avg losses for the epoch
    log_validation_losses_to_writer(metric_logger, global_step=epoch)

    return metric_logger.meters['loss'].global_avg


def extract_proposals_and_detections(model, device, data_loader, max_number_of_images):
    model.eval()
    cpu_device = torch.device("cpu")

    count_imgs = 0
    count_detections = 0
    outputs_per_img = []
    for images, targets in tqdm(data_loader):

        count_imgs += len(images)
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.no_grad():  # We are validating, we do not use the gradients
            detections = model(images, targets)

        for one_img_detection in detections:
            count_detections += 1
            outputs_per_img.append({k: v.to(cpu_device) for k, v in one_img_detection.items()})

        if count_imgs > max_number_of_images:
            break

    print(f'Number of images processed: {count_imgs}')
    print(f'Number of detections processed: {count_detections}')

    return outputs_per_img


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types


def compute_mean_avg_precision(model, device, data_loader, known_classes=None):
    print('Computing mean average precision for validation dataset')
    metric_logger = MetricLogger(delimiter="  ")
    header = "Validation:"
    model.eval()
    cpu_device = torch.device("cpu")

    # Define the object to evaluate
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    print(f'{coco_evaluator.coco_eval["bbox"].params.catIds}')

    time_elapsed = time.perf_counter()
    for images, targets in metric_logger.log_every(data_loader, 100, header):

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            outputs = model(images)
            if args.rm_bg:
                remove_bg_predictions(outputs)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"][0].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    time_elapsed = time.perf_counter() - time_elapsed
    print(f"Time spent generating detections: {time_elapsed}")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    print('Validating finished.')
    return coco_evaluator


def remove_bg_predictions(detections: List[Dict]):
    """
    Modifies the detections list (in-place) to remove the predictions of BG
    """
    for img_preds in detections:

        inds_no_bg = torch.where(img_preds['labels'] != 0)[0]

        for k, v in img_preds.items():
            img_preds[k] = v[inds_no_bg]


def plot_images(model, device, data_loader, classes_list, save_figs=False, output_dir=None, option=''):
    max_count = 40
    if save_figs:
        assert isinstance(output_dir, Path), 'The output directory of the dataset must be specified'
        output_dir = output_dir / 'detections/'
        output_dir.mkdir(exist_ok=True)  # Create the folder, but ignore the exception if created
        print(f"Saving the first {max_count} images of the validation set to {output_dir}")
    else:
        print(f"Start plotting images, stops when hitting control+C or reaches {max_count} images")

    # Set evaluation mode for enabling the model to output the detections
    model.eval()
    cpu_device = torch.device("cpu")

    count = 0
    if option == 'imgs_and_preds':

        for images, targets in data_loader:

            count += len(images)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = model(images)

            # If outputs is a tuple, it means they are the proposals and its scores
            # With this code we arrange them in a good manner to be ingested by the
            # following code
            if isinstance(outputs, tuple):
                list_outputs = []
                for i in range(len(outputs[0])):
                    list_outputs.append((outputs[0][i], outputs[1][i]))
                outputs = list_outputs

            if args.rm_bg:
                remove_bg_predictions(outputs)

            for image, output, target in zip(images, outputs, targets):
                H, W = image.shape[1:]
                # figsize = (constant * (Height Width relation) * n_cols, constant * n_rows)
                fig, axes = plt.subplots(
                    1, 2, figsize=(6 * (W/H) * 2, 6 * 1), tight_layout=True)
                image = image.to(cpu_device)
                img = create_img_with_bboxes(image, output, classes_list)
                axes[0].imshow(img.permute(1, 2, 0))
                axes[0].set_title("Image with predictions", y=0.99)
                axes[0].set_axis_off()
                img = create_img_with_bboxes(image, target, classes_list)
                axes[1].imshow(img.permute(1, 2, 0))
                axes[1].set_title("Image with ground truth", y=0.99)
                axes[1].set_axis_off()
                fig.savefig(output_dir / f'{str(target["image_id"][0].item()).zfill(12)}.png', dpi=250)
                plt.close()

            print(f'{count} images out of {max_count} processed')
            if count > max_count:
                print(f'{max_count} images saved or represented, exiting program!')
                break

    elif option == 'one_img_preds':

        for images, targets in data_loader:

            count += len(images)
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = model(images)

            if isinstance(outputs, tuple):
                list_outputs = []
                for i in range(len(outputs[0])):
                    list_outputs.append((outputs[0][i], outputs[1][i]))
                outputs = list_outputs

            if args.rm_bg:
                remove_bg_predictions(outputs)

            for image, output, target in zip(images, outputs, targets):
                H, W = image.shape[1:]
                image = image.to(cpu_device)
                plt.imshow(image.permute(1, 2, 0))
                plt.axis('off')
                plt.savefig(output_dir / f'{str(target["image_id"][0].item()).zfill(12)}.pdf')
                plt.close()

            print(f'{count} images out of {max_count} processed')
            if count > max_count:
                print(f'{max_count} images saved or represented, exiting program!')
                break

        else:
            raise NameError


def extract_spike_rates(model, device, data_loader):
    """
    Only works if the GeneralizedRCNN object and the diferent layers are hacked manually. All stuff that
    must be hacked is marked with ### EXTRACT SPIKE RATES ###. Steps to hack:
    1. Change the forward function of the RPNHeadSNN (from rpn.py) and the RoIHeadsSNN (from faster_rcnn.py)
        to the one indicated with # Extract spike rates by commenting the other forward function
    2. Edit both RegionProposalNetwork (from rpn.py) and RoIHeadsSNN (from roi_heads.py) to return immediately
        the spike rates after computing them to skip all the unnecessary code of the transformations
    3. In GeneralizedRCNN, just uncomment ### EXTRACT SPIKE RATES ### part
    """
    model.eval()
    cpu_device = torch.device("cpu")

    count_imgs = 0
    count_detections = 0
    spike_rates_per_img = []
    from collections import defaultdict
    all_images_per_layer_dict = defaultdict(list)
    for images, targets in tqdm(data_loader):

        count_imgs += len(images)
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.no_grad():  # We are validating, we do not use the gradients
            spike_rates_few_images = model(images, targets)

        for i, few_imgs_spk_rate in enumerate(spike_rates_few_images):
            all_images_per_layer_dict[i].append(few_imgs_spk_rate.to(cpu_device))

        #spike_rates_per_img.append(item.to(cpu_device) for item in spike_rates_few_images)

    # from collections import defaultdict
    # all_images_per_layer_dict = defaultdict(list)
    # for one_batch_of_spikes in spike_rates_per_img:
    #     for i, one_layer_spk_rate in enumerate(one_batch_of_spikes):
    #         all_images_per_layer_dict[i].append(one_layer_spk_rate)

    for k in all_images_per_layer_dict.keys():
        all_images_per_layer_dict[k] = torch.cat(all_images_per_layer_dict[k], dim=0)
        print(all_images_per_layer_dict[k].shape)
    # spike_rates_per_img = torch.stack(spike_rates_per_img, dim=0)

    print(f'Number of images processed: {count_imgs}')

    flops_per_layer = []
    rpn_layers = ['LVL_0', 'LVL_1', 'LVL_2' ,'LVL_3', 'pool']
    rpn_layers_counter = 0
    detector_layers = ['FC6', 'FC7']
    detector_layers_counter = 0
    timesteps_rpn = model.rpn.head.num_steps
    timesteps_detector = model.roi_heads.box_head_and_predictor.num_steps
    for k, v in all_images_per_layer_dict.items():

        # RPN
        if k in [0, 3, 6, 9, 12]:  # Those are the positions of the layers with spikes

            mean_spk_per_layer_per_image = v[:, 0].mean() * timesteps_rpn
            flops_layer = v[0, 1]
            flops_per_layer.append([mean_spk_per_layer_per_image, flops_layer])
            print(f'{rpn_layers[rpn_layers_counter]}:\tNumero medio spikes en {timesteps_rpn} timesteps (RPN): {mean_spk_per_layer_per_image.item():.4f}')
            rpn_layers_counter += 1

        # Detector
        if k in [15, 16]:  # Those are the positions of the layers with spikes

            mean_spk_per_layer_per_bbox = v[:, 0].mean() * timesteps_detector
            flops_layer = v[0, 1] * 1000  # X1000 because it is calculated per bbox and 1000 bboxes are used
            # Other way to get that number is to take the lenght of the v (len(v) or v.shape[0])
            flops_per_layer.append([mean_spk_per_layer_per_bbox, flops_layer])
            print(f'{detector_layers[detector_layers_counter]}:\tNumero medio spikes en {timesteps_detector} timesteps (Det): {mean_spk_per_layer_per_bbox.item():.4f}')
            detector_layers_counter += 1

    print()

    all_layers_names = rpn_layers + detector_layers
    ann_total_energy_consumption = 0
    snn_total_energy_consumption = 0
    for i, f in enumerate(flops_per_layer):
        ann_one_layer_energy_consumption = f[1] * 4.6 * 10 ** -12
        snn_one_layer_energy_consumption = f[0] * f[1] * 0.9 * 10 ** -12
        print(
            f'{all_layers_names[i]}:\tEnergía ANN:\t{ann_one_layer_energy_consumption:.5f} | Energía SNN:\t{snn_one_layer_energy_consumption:.5f} '
            f'| Reduccion de consumo: {((f[0] * f[1] * 0.9) / (f[1] * 4.6)) * 100:.2f}%'
        )
        ann_total_energy_consumption += ann_one_layer_energy_consumption
        snn_total_energy_consumption += snn_one_layer_energy_consumption

    print(f'Reduccion de consumo total: {(snn_total_energy_consumption/ann_total_energy_consumption)*100:.2f}%')

    return all_images_per_layer_dict

def main(args):

    # -----------------------------
    # Load all configuration
    # -----------------------------
    # For saving the images, we want to enable the code that leads
    # to the function plot_images
    if args.save_images:
        args.plot_images = True
    if args.test_only:
        args.validate_each_epoch = True
    # For the option of plotting images, we only want to load
    # the validation dataset, so we can config args.test_ony
    if args.plot_images:
        args.distributed = False
        args.test_only = True
    else:
        # Initiate distributed mode if requested
        init_distributed_mode(args)
    print('| *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* |')
    print('|                                                                                             |')
    print('|                              STARTING Spiking FASTER R-CNN                                  |')
    print('|                                                                                             |')
    print('| *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* |')
    print(args)

    # Load config. Added datasets must have same variables defined
    CONFIG = load_config_dict(args.dataset)
    try:
        args.output_dir = Path(CONFIG["OUT_DIR"])
        NUM_CLASSES = CONFIG["NUM_CLASSES"]
    except KeyError as e:
        raise KeyError(f'Check that the keys in the .yaml files of config are correct: {e}')

    # If we have defined known class, we need to change the known classes var
    known_classes = None
    if args.only_known_cls:
        try:
            known_classes = CONFIG['KNOWN_CLASSES']
            NUM_CLASSES = len(known_classes)
        except KeyError:
            pass

    # Created the folder in case it does not exist
    check_folder_exist(args.output_dir)

    # -----------------------------
    # Data loading definition
    # -----------------------------
    # Create the train and validation sets

    if not args.test_only:
        train_dataset = create_dataset(
            args.dataset, set_option='train',
            transforms=get_img_transform_coco(train=True, data_augm=args.data_augmentation),
            known_classes=known_classes
        )

    if args.add_noise:
        val_data_augmentation = args.add_noise
    else:
        val_data_augmentation = args.data_augmentation
    val_dataset = create_dataset(
        args.dataset,
        set_option='validation',
        transforms=get_img_transform_coco(
            train=False, data_augm=val_data_augmentation, noise_intensity=args.noise_intensity
        ),
        known_classes=known_classes
    )

    if 'all' in args.extract_proposals_and_detections:
        args.extract_proposals_and_detections = ['train', 'test']

    # Create the samplers
    if args.distributed:
        if not args.test_only:  # ORIGINAL SEED:15
            if 'train' in args.extract_proposals_and_detections:
                # If we want to extract spike counts from train, we want the order to be always the same
                train_sampler = distributed.DistributedSampler(train_dataset, shuffle=False)
            else:
                train_sampler = distributed.DistributedSampler(train_dataset, shuffle=True, seed=12)
        val_sampler = distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        # g = torch.Generator()
        # g.manual_seed(0)
        if not args.test_only:
            train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

    # if not args.test_only:
    #     if args.aspect_ratio_group_factor >= 0:
    #         group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
    #         train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    #     else:
    #         train_batch_sampler = BatchSampler(train_sampler, args.batch_size, drop_last=True)

    # Create the loaders
    if not args.test_only:
        train_loader = create_data_loader(train_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=args.workers,
                                          sampler=train_sampler,
                                          shuffle=False)
        print(f"Number of training samples: {len(train_dataset)}")
    val_loader = create_data_loader(val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=args.workers,
                                    sampler=val_sampler,
                                    shuffle=False)

    print(f"Number of validation samples: {len(val_dataset)}\n")

    # -----------------------------
    # Model definition
    # -----------------------------
    model = create_model(
        dataset_name=args.dataset, num_classes=NUM_CLASSES, rpn_snn=args.rpn_snn, detector_snn=args.detector_snn,
        trainable_backbone_layers=args.trainable_backbone_layers, pretrained_rpn_and_detector=args.pretrained,
        pretrained_fpn=args.pretrained_fpn, only_one_bbox=args.only_one_bbox,
        num_steps_rpn=args.num_steps_rpn, num_steps_detector=args.num_steps_detector
    )
    if args.distributed:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = args.device if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    # -----------------------------
    # Load model and freeze parts
    # -----------------------------
    if args.load_model:
        if args.resume:
            raise NotImplementedError('Currently not implemented the posibility to resume RPN training')
        else:
            print('----------------------------------')
            print('       LOADING WEIGHTS')
            args.load_model = Path(args.load_model)  # args.resume must contain the path to the checkpoint
            try:
                checkpoint = torch.load(args.load_model, map_location="cpu")
            except NotImplementedError:
                print('')
                print('WARNING: Loading Backbone and RPN saved on Linux into a Windows machine', \
                      'overriding pathlib.PosixPath with pathlib.WindowsPath to enable the load')
                print('')
                import pathlib
                temp = pathlib.PosixPath
                pathlib.PosixPath = pathlib.WindowsPath
                checkpoint = torch.load(args.load_model, map_location="cpu")

            if args.only_one_bbox:
                shape_bbox_pred = checkpoint["model"]['roi_heads.box_head_and_predictor.bbox_pred.weight'].size(dim=0)
                if shape_bbox_pred != NUM_CLASSES:  # It means we are loading model with more than one bbox
                    print('Popping the "roi_heads.box_head_and_predictor.bbox_pred.weight" from the checkpoint')
                    checkpoint["model"].pop('roi_heads.box_head_and_predictor.bbox_pred.weight')
            print(model.load_state_dict(checkpoint["model"], strict=False))
            print('----------------------------------')

    # It is important to freeze parameters BEFORE wrapping the model with the DistributedDataParallel(),
    # otherwise it would have errors
    if args.freeze_fpn:
        print('')
        print('--  Freezing Feature Pyramid Network  --')
        print('')
        for p in model.backbone.named_parameters():
            if p[0].startswith('fpn'):
                p[1].requires_grad = False

    if args.freeze_rpn:
        print('')
        print('--  Freezing RPN  --')
        print('')
        for p in model.rpn.parameters():
            p.requires_grad = False
        #model.rpn.training = False

    if args.freeze_detector:
        print('')
        print('--  Freezing Detector  --')
        print('')
        for p in model.roi_heads.parameters():
            p.requires_grad = False

    # Distributed. device_ids: gpu id that the model lives on
    model_without_ddp = model
    if args.distributed:
        # find_unused_parameters, if set to True, generates performance issues, as it is used to avoid errors
        # when calling to reduce dict across processes. This errors are the ones generated when a part of the model
        # is unused.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False
            )
        model_without_ddp = model.module

    # -----------------------------
    # Optimizer definition
    # -----------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    # Learning rate scheduler
    if args.lr_decay_milestones:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.lr_decay_milestones,
            gamma=args.lr_decay_rate,
            last_epoch=-1,
            verbose=True
        )
    elif args.lr_decay_step:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate, verbose=True
        )
    else:
        print('No LR scheduler used')

    if args.constant_lr_scheduler:
        print('Using the ConstantLR to adjust by a factor the first epoch of the training')
        lr_scheduler = [lr_scheduler, torch.optim.lr_scheduler.ConstantLR(
            optimizer=optimizer,
            factor=args.constant_lr_scheduler,
            total_iters=1,
            verbose=True
        )]

    # Gradient scaling for improved memory usage
    # https://pytorch.org/docs/stable/amp.html#gradient-scaling
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # -----------------------------
    # Resuming checkpoint
    # -----------------------------
    if args.resume:
        print('-----------------------------')
        print('         RESUMING TRAINING        ')
        args.resume = Path(args.resume)  # args.resume must contain the path to the checkpoint
        try:
            checkpoint = torch.load(args.resume, map_location="cpu")
        except NotImplementedError:
            print('')
            print('WARNING: Loading model saved on Linux into a Windows machine', \
                  'overriding pathlib.PosixPath with pathlib.WindowsPath to enable the load')
            print('')
            import pathlib
            temp = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            checkpoint = torch.load(args.resume, map_location="cpu")
    
        print(model_without_ddp.load_state_dict(checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"]
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
        print("Loading checkpoint finished")
        print('-----------------------------')

    # -----------------------------
    # Only plotting images
    # -----------------------------
    if args.plot_images:
        if args.only_known_cls:
            classes = known_classes
        else:
            classes = CONFIG["CLASSES"]
        plot_images(model, device, val_loader, classes,
                    save_figs=args.save_images, output_dir=args.output_dir, option=args.save_images_option)
        return

    # -----------------------------
    # Extract proposals and detections
    # -----------------------------
    for split in args.extract_proposals_and_detections:

        if split == "train":
            print('Extracting TRAIN proposals and detections')
            train_results_per_img = extract_proposals_and_detections(
                model, device, train_loader, args.max_num_images_for_nod
            )
            torch.save(
                train_results_per_img, args.output_dir / f'train_results_per_img_{args.dataset}.pt'
            )

        elif split == "test":
            print('Extracting TEST (or validation) proposals and detections')
            test_results_per_img = extract_proposals_and_detections(
                model, device, val_loader, args.max_num_images_for_nod
            )
            torch.save(
                test_results_per_img, args.output_dir / f'test_results_per_img_{args.dataset}.pt'
            )

        else:
            print(f'WARNING: Wrong option selected: {split}. Skipping iteration')

    if args.extract_proposals_and_detections:
        # Finish the script if we have extracted proposals and detections
        print('*-'*25 + '*')
        print()
        print(f'Finished extracting proposals and detections, you can find them in the output directory: {args.output_dir}')
        print()
        print('*-' * 25 + '*')
        return

    # -----------------------------
    # Extract spike rates
    # -----------------------------
    for split in args.extract_spike_rates:

        if split == "train":
            print('Extracting TRAIN spike counts')
            train_spike_rates_per_img = extract_spike_rates(model, device, train_loader)
            torch.save(train_spike_rates_per_img, args.output_dir / f'train_spike_rates_per_img_{args.load_model.stem}.pt')
        elif split == "test":
            print('Extracting TEST (or validation) spike counts')
            test_spike_rates_per_img = extract_spike_rates(model, device, val_loader)
            torch.save(test_spike_rates_per_img, args.output_dir / f'test_spike_rates_per_img_{args.load_model.stem}.pt')

        else:
            print(f'WARNING: Wrong option selected: {split}. Skipping iteration')

    if args.extract_spike_rates:
        # Finish the script if we have extracted spike counts
        print('*-'*25 + '*')
        print()
        print(f'Finished extracting spike rates, you can find them in the output directory: {args.output_dir}')
        print()
        print('*-' * 25 + '*')
        return

    # If we only want to test, doing several epochs is not needed
    if args.test_only:
        args.start_epoch = 1
        args.epochs = 2

    check_freezed_modules(model_without_ddp)
    start = time.perf_counter()

    # -----------------------------
    # Testing
    # -----------------------------
    if args.test_only:
        if args.distributed:
            if args.dist_backend == "gloo":
                evaluator = compute_mean_avg_precision(model, device, val_loader, known_classes=known_classes)
            else:
                print("---------------------------------------------------------")
                print("Skipping evaluator with NCCL backend till its debugged")
                print("---------------------------------------------------------")
        else:
            evaluator = compute_mean_avg_precision(model, device, val_loader, known_classes=known_classes)
        print(f"Took {((time.perf_counter() - start) / 60):.3f} minutes to compute mAP on validation data")

    # -----------------------------
    # Training loop
    # -----------------------------
    else:
        print_hyperparameters(model_without_ddp, optimizer, lr_scheduler, args)
        # +1 is to reach the number of desired epochs, as range function does not include the last number
        for epoch in range(args.start_epoch, args.epochs + 1):

            print(f"\nEPOCH {epoch} of {args.epochs}")
            # Metric logger will be the iterable to loop over in each epoch, and is in charge of
            # printing statistics every args.print_freq batches processed
            # Every epoch is restarted to show the global metrics for the current epoch
            # metric_logger = MetricLogger(delimiter="  ")
            start_time_epoch = time.perf_counter()

            if args.distributed:
                # When :attr:`shuffle=True`, this ensures all replicas
                # use a different random ordering for each epoch.
                train_sampler.set_epoch(epoch)

            # Start timer and carry out training and validation
            print("Start training")
            training_global_avg_loss = train_one_epoch(
                model, device, train_loader, optimizer, epoch, args.print_freq, scaler
            )
            print(f"Took {((time.perf_counter() - start_time_epoch) / 60):.3f} minutes for training one epoch")
            print('***************************************')
            print(f"Average loss of epoch #{epoch}: {training_global_avg_loss}")
            print('***************************************')

            # update the learning rate
            if lr_scheduler:
                if isinstance(lr_scheduler, list):
                    for sched in lr_scheduler:
                        sched.step()
                else:
                    lr_scheduler.step()

            # Save model
            if args.output_dir:
                if isinstance(lr_scheduler, list):
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler[0].state_dict(),
                        "args": args,
                        "epoch": epoch,
                    }
                else:
                    checkpoint = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "args": args,
                        "epoch": epoch,
                    }
                if args.amp:
                    checkpoint["scaler"] = scaler.state_dict()
                print("Saving weights...")
                if args.save_name:
                    save_on_master(checkpoint, args.output_dir / f"model_{args.save_name}_{int(epoch)}.pth")
                else:
                    save_on_master(checkpoint, args.output_dir / f"model_{int(epoch)}.pth")
                save_on_master(checkpoint, args.output_dir / "checkpoint.pth")
                print(f"Weights saved for epoch {epoch} and also the checkpoint")
            else:
                print('WARNING: No output directory specified in the config file, skipping saving weights!')

            # Validation only every N epochs or the last epoch (where epoch == number of epochs specified in args)
            if epoch % args.validate_every_n_epochs == 0 or epoch == args.epochs:
                start_val = time.perf_counter()
                validation_global_avg_loss = validate_one_epoch(
                    model, device, val_loader, epoch, args.print_freq, scaler
                )
                print(f"Took {((time.perf_counter() - start_val) / 60):.3f} minutes for validating one epoch")
                print('***************************************')
                print(f"Average loss of epoch #{epoch}: {validation_global_avg_loss}")
                print('***************************************')
            else:
                print('Skipping validation as specified in the arguments of the call')

                print(f"Took {((time.perf_counter() - start_time_epoch) / 60):.3f} minutes for the complete "
                      f"epoch number {epoch}")

        print('Training finished')
        print(f"Took {((time.perf_counter() - start) / 60):.3f} minutes for all training")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    # Globals
    if not args.test_only and not args.save_images and not args.extract_proposals_and_detections:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        if args.save_name:
            train_writer = SummaryWriter(f'runs/{timestamp}_Train_{args.save_name}')
            val_writer = SummaryWriter(f'runs/{timestamp}_Validation_{args.save_name}')
        else:
            train_writer = SummaryWriter(f'runs/{timestamp}_Train')
            val_writer = SummaryWriter(f'runs/{timestamp}_Validation')
    main(args)
