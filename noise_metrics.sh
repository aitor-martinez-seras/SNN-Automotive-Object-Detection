#!/bin/bash

python noise_calculations.py -d cityscapes -b 2 --rpn-snn --detector-snn --load-model outputs/cityscapes/model_Cityscapes_SNN.pth
python noise_calculations.py -d cityscapes -b 2 --rpn-snn --detector-snn --load-model outputs/cityscapes/model_Cityscapes_SNN_FPN_tuned.pth
python noise_calculations.py -d cityscapes -b 2 --load-model outputs/cityscapes/model_Cityscapes_NoSNN.pth
python noise_calculations.py -d cityscapes -b 2 --load-model outputs/cityscapes/model_Cityscapes_NoSNN_FPN_tuned.pth

python noise_calculations.py -d cityscapes -b 2 --rpn-snn --detector-snn --load-model outputs/cityscapes/model_Cityscapes_SNN.pth --rain-noise
python noise_calculations.py -d cityscapes -b 2 --rpn-snn --detector-snn --load-model outputs/cityscapes/model_Cityscapes_SNN_FPN_tuned.pth --rain-noise
python noise_calculations.py -d cityscapes -b 2 --load-model outputs/cityscapes/model_Cityscapes_NoSNN.pth --rain-noise
python noise_calculations.py -d cityscapes -b 2 --load-model outputs/cityscapes/model_Cityscapes_NoSNN_FPN_tuned.pth --rain-noise

python noise_plots.py