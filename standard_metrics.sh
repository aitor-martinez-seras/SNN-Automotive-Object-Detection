#!/bin/bash

python train.py -d cityscapes -b 2 --rpn-snn --detector-snn --load-model /outputs/cityscapes/model_cityscapes_SNN.pth --test-only
python train.py -d cityscapes -b 2 --rpn-snn --detector-snn --load-model /outputs/cityscapes/model_cityscapes_SNN_FPN_tuned.pth --test-only
python train.py -d cityscapes -b 2 --load-model /outputs/cityscapes/model_cityscapes_NoSNN.pth --test-only
python train.py -d cityscapes -b 2 --load-model /outputs/cityscapes/model_cityscapes_NoSNN_FPN_tuned.pth --test-only

python train.py -d bdd -b 2 --rpn-snn --detector-snn --load-model /outputs/bdd/model_BDD_SNN.pth --test-only
python train.py -d bdd -b 2 --rpn-snn --detector-snn --load-model /outputs/bdd/model_BDD_SNN_FPN_tuned.pth --test-only
python train.py -d bdd -b 2 --load-model /outputs/bdd/model_BDD_NoSNN.pth --test-only
python train.py -d bdd -b 2 --load-model /outputs/bdd/model_BDD_NoSNN_FPN_tuned.pth --test-only

python train.py -d idd -b 2 --rpn-snn --detector-snn --load-model /outputs/idd/model_IDD_SNN.pth --test-only
python train.py -d idd -b 2 --rpn-snn --detector-snn --load-model /outputs/idd/model_IDD_SNN_FPN_tuned.pth --test-only
python train.py -d idd -b 2 --load-model /outputs/idd/model_IDD_NoSNN.pth --test-only
python train.py -d idd -b 2 --load-model /outputs/idd/model_IDD_NoSNN_FPN_tuned.pth --test-only