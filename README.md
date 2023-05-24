# OWOD_SNN

This is the repository containing the code and results for the paper ____. The paper is currently being evaluated for the ITSC 2023 Obtained results are available in the results folder.

## Requirements

Requierements used:

- norse 0.0.7
- torch 1.12.1 + cu113
- torchaudio 0.12.1
- torchvision 0.13.1
- pycocotools
- PyYAML
- Common science python kit (numpy, pandas, scypy, sklearn)
- Tensorboard

## Installation guide

It is recommended to use a virtualenv. It can be installed also in conda environments.

1. Install PyTorch desired GPU version (<https://pytorch.org/get-started/previous-versions>), checking compatibility with Norse (<https://github.com/norse/norse>)
2. Install Norse. The recommended way is to clone the repo and then run ````python setup.py install````.
In case of problems with norse, refer to
[installation troubleshooting of norse](https://norse.github.io/norse/pages/installing.html#installation-troubleshooting).
3. Install the other dependencies.

## Datasets

### Cityscapes

To download the dataset, visit the [official website](https://www.cityscapes-dataset.com/), register, and download the ```leftImg8bit_trainvaltest.zip``` file, that can be found in the
[downloads](https://www.cityscapes-dataset.com/downloads/)
section. Extract its content in _data/cityscapes_ folder.

The annotations are automatically downloaded from this [repository](https://tecnalia365-my.sharepoint.com/:u:/g/personal/aitor_martinez_tecnalia_com/EfD21vmwQztJpp_Rg8nB9ecBkKNM3a1uV8ekVeU4TP8OTw?download=1) to _data/annotations_.

### Berkeley DeepDrive (BDD)

Download the images from the [official website](https://bdd-data.berkeley.edu/portal.html#download) clicking on the button "100k Images".

The labels are downloaded automatically from the cloud, as they have been converted into coco format manually.

### Indian Driving Dataset (IDD)

Download the dataset from the [official website](https://idd.insaan.iiit.ac.in/dataset/download/) (Dataset Name: IDD Detection) and extract the content of the downloaded file to ````data/idd/````.

## Reproduce results

First download all models from the [following link](OneDriveLink) and include them in the desired folder. We recomend ````outputs/<dataset name>/````.

### Metrics for the standard models

Run

````shell
standard_metrics.sh
````

### Noise results

Run

````shell
noise_metrics.sh
````

### Efficiency results

Precision and recall metrics:

````shell
python test_and_energy_eff.py -d cityscapes -b 2  --rpn-snn --detector-snn --load-model outputs/cityscapes/model_Cityscapes_SNN_Trpn8_Tdet12.pth --test-only -o metrics
````

Efficiency w.r.t the NoSNN models. For this computation the following modifications on the source code are needed. All changes are marked in the source code with the flag ````### EXTRACT SPIKE RATES ###````. It can be followed by the word "activate" or "deactivate":

- If ````### EXTRACT SPIKE RATES ###```` activate, then uncomment that block for the computations
- If ````### EXTRACT SPIKE RATES ###```` deactivate, then comment that block for the computations

1. Change the forward function of the RPNHeadSNN (from rpn.py) and the RoIHeadsSNN (from faster_rcnn.py) to the one indicated with the flag (comment the normal forward function)
2. Edit both RegionProposalNetwork (from rpn.py) and RoIHeadsSNN (from roi_heads.py) to return immediately the spike rates after computing them to skip all the unnecessary code of the transformations. Again, look for the flag. In rpn.py, 2 blocks of the RegionProposalNetwork object have to be activated and 1 deactivated. In roi_heads.py only one block has to be activated.
3. In GeneralizedRCNN, just uncomment the part indicated with the flag.

````shell
python test_and_energy_eff.py -d cityscapes -b 2  --rpn-snn --detector-snn --load-model outputs/cityscapes/model_Cityscapes_SNN_Trpn8_Tdet12.pth --test-only -o efficiency
````

### New object discovery

First it is necessary to extract the proposals and detections from the images. With the ```-n-img``` argument the number of images retrieved can be modified:

Cityscapes

````shell
python train.py -d bdd --batch-size 4  --rpn-snn --detector-snn --load-model outputs/bdd/model_BDD_SNN_5cls.pth --only-known-cls -ext-prop-det test -n-img 500
````

BDD

````shell
python train.py -d bdd --batch-size 4  --rpn-snn --detector-snn --load-model outputs/bdd/model_BDD_SNN_5cls.pth --only-known-cls -ext-prop-det test -n-img 2000
````

To plot the images:

Cityscapes

````shell
python new_object_discovery.py -d cityscapes -f outputs/cityscapes/test_results_per_img_cityscapes.pt --score-thr 2 --nms-thr 0.25 -max 6 --save-images 50 --only-known-cls --iou-thr 0 --plot-all-in-one-img
````

BDD

````shell
python new_object_discovery.py -d bdd -f outputs/cityscapes/test_results_per_img_bdd.pt --score-thr 2 --nms-thr 0.25 -max 6 --save-images 50 --only-known-cls --iou-thr 0 --plot-all-in-one-img
````

## Usage

### Training

Following the training protocol exposed in the paper, we first train the RPN, then the detector, and finally tune all together with the FPN. Example with Cityscapes:

Training RPN:

````shell
python train.py -d cityscapes -b 2 --epochs 25 --lr-decay-rate 0.5 --lr 0.0005 --lr-decay-milestones 10 15 20 --rpn-snn --detector-snn --freeze-fpn --freeze-detector
````

Training Detector:

````shell
python train.py -d cityscapes -b 2 --epochs 25 --lr-decay-rate 0.5 --lr 0.0005 --lr-decay-milestones 10 15 20 --rpn-snn --detector-snn --freeze-fpn --freeze-rpn --load-model /outputs/cityscapes/path-to-the-model
````

Finetuning FPN:

````shell
python train.py -d cityscapes -b 2 --epochs 15 --lr-decay-rate 0.5 --lr 0.00005 --lr-decay-milestones 5 10 --rpn-snn --detector-snn --load-model /outputs/cityscapes/path-to-the-model
````

To train only for certain amount of classes, add ```--only-known-cls``` and the classes indicated as known in the config file of the dataset will be used.

### Testing

Obtain the metrics of the trained model

````shell
python train.py -d cityscapes -b 2 --rpn-snn --detector-snn --load-model /outputs/cityscapes/path-to-the-model --test-only
````

To add noise to testing

````shell
python train.py -d cityscapes -b 2 --rpn-snn --detector-snn --load-model /outputs/cityscapes/path-to-the-model --test-only --add-noise
````

To test only for certain amount of classes, add ```--only-known-cls``` and the classes indicated as known in the config file of the dataset will be used.
