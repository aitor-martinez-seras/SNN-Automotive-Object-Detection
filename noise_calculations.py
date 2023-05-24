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
import json
from typing import List, Dict

import torch
import torch.utils.data.distributed as distributed
from torch.utils.data import SequentialSampler

from datasets import create_dataset, create_data_loader, get_img_transform_coco
from model import create_model
from custom_utils import (
    check_folder_exist, load_config_dict,
)
from utils import MetricLogger, init_distributed_mode
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch SNN Detection Training", add_help=add_help)

    parser.add_argument("-d", "--dataset", default="cityscapes", type=str, required=True, help="dataset name",
                        choices=['cityscapes', 'bdd', 'idd'])
    parser.add_argument("-t-rpn", "--rpn-steps", default=12, dest='num_steps_rpn', type=int,
                        help="number of total steps of the RPN")
    parser.add_argument("-t-det", "--det-steps", default=16, dest='num_steps_detector', type=int,
                        help="number of total steps of the detector")
    parser.add_argument("-r1", "--rpn1", default=4, type=int)
    parser.add_argument("-r2", "--rpn2", default=12, type=int)
    parser.add_argument("-d1", "--det1", default=8, type=int)
    parser.add_argument("-d2", "--det2", default=16, type=int)

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
    parser.add_argument("--plot-images", dest="plot_images", action="store_true", default=False,
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
    parser.add_argument("--data-augmentation", dest="data_augmentation", default="hflip", type=str,
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
    parser.add_argument("--extract-spike-rates", nargs="+", default=[], dest="extract_spike_rates",
                        help="Pass train and/or test to extract spike rates and FLOPs of "
                             "train and/or test data and save to a file")
    parser.add_argument("--rain-noise", default=False, action="store_true", dest="rain_noise",
                        help="Use rain noise")

    return parser


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
        # model.rpn.training = False

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
    # Data loading definition
    # -----------------------------
    if args.rain_noise:
        save_name = args.output_dir / f'rain_noise_acc_{Path(args.load_model).stem}.json'
        all_results = []
        experiments_number_of_drops = [x for x in range(0, 301, 50)]
        for idx, n_rain_drops in enumerate(experiments_number_of_drops):
            print(f'+ - + - + - + - + - + - + - + - + - + - + - + - +')
            print(f'Number of raindrops = {n_rain_drops}')
            print(f'+ - + - + - + - + - + - + - + - + - + - + - + - +')
            val_dataset = create_dataset(
                args.dataset,
                set_option='validation',
                transforms=get_img_transform_coco(
                    train=False, data_augm='rain', noise_intensity=n_rain_drops
                ),
                known_classes=known_classes
            )

            # Create the samplers
            if args.distributed:
                val_sampler = distributed.DistributedSampler(val_dataset, shuffle=False)
            else:
                # g = torch.Generator()
                # g.manual_seed(0)
                val_sampler = SequentialSampler(val_dataset)

            val_loader = create_data_loader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
                sampler=val_sampler,
                shuffle=False
            )

            # -----------------------------
            # Compute results
            # -----------------------------
            coco_evaluator_one_iter = compute_mean_avg_precision(model, device, val_loader)
            results = coco_evaluator_one_iter.coco_eval['bbox'].stats
            # precision[idx_t_rpn, idx_t_det]
            all_results.append([n_rain_drops, results[0], results[1], results[8]])
            print(all_results)

            with open(save_name, "w") as fp:
                json.dump(all_results, fp)
    else:
        save_name = args.output_dir / f'noise_acc_{Path(args.load_model).stem}.json'
        all_results = []
        intensity_experiments = [x * 0.01 for x in range(25)]
        noise = 'gaussian'
        for idx, intensity in enumerate(intensity_experiments):
            print(f'+ - + - + - + - + - + - + - + - + - + - + - + - +')
            print(f'Noise type      = {noise}')
            print(f'Noise intensity = {intensity}')
            print(f'+ - + - + - + - + - + - + - + - + - + - + - + - +')
            val_dataset = create_dataset(
                args.dataset,
                set_option='validation',
                transforms=get_img_transform_coco(
                    train=False, data_augm=noise, noise_intensity=intensity
                ),
                known_classes=known_classes
            )

            # Create the samplers
            if args.distributed:
                val_sampler = distributed.DistributedSampler(val_dataset, shuffle=False)
            else:
                # g = torch.Generator()
                # g.manual_seed(0)
                val_sampler = SequentialSampler(val_dataset)

            val_loader = create_data_loader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
                sampler=val_sampler,
                shuffle=False
            )

            # -----------------------------
            # Compute results
            # -----------------------------
            coco_evaluator_one_iter = compute_mean_avg_precision(model, device, val_loader)
            results = coco_evaluator_one_iter.coco_eval['bbox'].stats
            all_results.append([noise, intensity, results[0], results[1], results[8]])
            print(all_results)

            with open(save_name, "w") as fp:
                json.dump(all_results, fp)

    # Finish the script if we have extracted spike counts
    print(all_results)
    print('*-' * 25 + '*')
    print()
    print(f'Finished extracting spike rates, you can find them in the output directory: {args.output_dir}')
    print()
    print('*-' * 25 + '*')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
