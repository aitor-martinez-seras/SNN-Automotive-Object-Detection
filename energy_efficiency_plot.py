from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch SNN energy efficiency", add_help=add_help)

    parser.add_argument("-f", "--file", type=str, help="path to the file with the detections")
    parser.add_argument("-t-rpn", "--rpn-steps", default=12, dest='num_steps_rpn', type=int,
                        help="number of total steps of the RPN")
    parser.add_argument("-t-det", "--det-steps", default=16, dest='num_steps_detector', type=int,
                        help="number of total steps of the detector")
    parser.add_argument("-p", "--plot", action='store_true', help="plot")

    return parser


def main(args):

    if args.plot:
        with open(r"outputs/cityscapes/efficiency_model_Cityscapes_SNN_Trpn8_Tdet12.json", "r") as fp:
            results_eff = json.load(fp)
        with open(r"outputs/cityscapes/metrics_model_Cityscapes_SNN_Trpn8_Tdet12.json", "r") as fp:
            results_perf = json.load(fp)

        # Efficiency (% of ANN consumption) and performance
        trpn = np.array([x[0] for x in results_eff])
        tdet = np.array([x[1] for x in results_eff])
        efficiency = np.array([x[2] for x in results_eff]) * 100
        map05_perf = np.array([x[3] for x in results_perf]) * 100
        map05095_perf = np.array([x[2] for x in results_perf]) * 100
        mar_perf = np.array([x[4] for x in results_perf])

        # Create an score that is the tradeoff between eff and perf
        max_consumption, min_consumption = efficiency.max(), efficiency.min()
        max_perf, min_perf = map05_perf.max(), map05_perf.min()
        normalized_consumption = (efficiency - min_consumption) / (max_consumption - min_consumption)
        normalized_perf = (map05_perf - min_perf) / (max_perf - min_perf)
        normalized_eff = np.abs(1 - normalized_consumption)

        # Control de tradeoff weights with W
        w_eff = 1
        w_perf = 1
        tradeoff = w_eff * normalized_eff + w_perf * normalized_perf

        # Construct the matrix
        tradeoff_matrix = np.zeros((9, 9))
        for i, t in enumerate(tradeoff):
            tradeoff_matrix[8 - results_eff[i][0] + 4, 8 - results_eff[i][1] + 8] = t

        # Define the params
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
        })

        # Plot the tradeoff as a matrix
        im = plt.imshow(tradeoff_matrix, cmap='RdYlGn')  # im.axes.dataLim has the size of the image

        general_string = r"$({eff:.1f} \%, {map:.1f} \%)$"
        # general_string = "$({eff:.1f}, {map:.1f})$"

        edge_color = "blue"
        point_color = "blue"
        font_size = 14
        # Mid point
        plt.plot(4, 4, marker="o", markersize=10, markeredgecolor=edge_color, markerfacecolor=point_color)
        text = general_string.format(eff=results_eff[40][2] * 100, map=results_perf[40][3] * 100)
        plt.text(4.2, 3.9, s=r"\textbf{" + text + "}", fontdict={"fontsize": font_size})
        # plt.text(4.2, 3.9, s=general_string.format(eff=results_eff[40][2] * 100, map=results_perf[40][3] * 100), fontdict={"fontsize": font_size})
        # plt.text(4.2, 3.9, s=text, fontdict={"fontsize": 12})

        # Bottom left
        plt.plot(0, 8, marker="o", markersize=10, markeredgecolor=edge_color, markerfacecolor=point_color)
        plt.text(0.2, 7.7, s=general_string.format(eff=results_eff[0][2] * 100, map=results_perf[0][3] * 100), fontdict={"fontsize": font_size})

        # Bottom right
        plt.plot(8, 8, marker="o", markersize=10, markeredgecolor=edge_color, markerfacecolor=point_color)
        plt.text(5, 7.7, s=general_string.format(eff=results_eff[8][2] * 100, map=results_perf[8][3] * 100), fontdict={"fontsize": font_size})

        # Top left
        plt.plot(0, 0, marker="o", markersize=10, markeredgecolor=edge_color, markerfacecolor=point_color)
        plt.text(0.2, 0.55, s=general_string.format(eff=results_eff[72][2] * 100, map=results_perf[72][3] * 100), fontdict={"fontsize": font_size})

        # Top right
        plt.plot(8, 0, marker="o", markersize=10, markeredgecolor=edge_color, markerfacecolor=point_color)
        plt.text(5, 0.55, s=general_string.format(eff=results_eff[80][2] * 100, map=results_perf[80][3] * 100), fontdict={"fontsize": font_size})

        ticks = [x for x in range(9)]
        det_labels = [x + 8 for x in range(9)]
        rpn_labels = [x + 4 for x in range(9)]
        rpn_labels.reverse()
        plt.xticks(ticks=ticks, labels=det_labels)
        plt.yticks(ticks=ticks, labels=rpn_labels)
        plt.xlabel('$T_{det}$')
        plt.ylabel('$T_{rpn}$')
        plt.savefig('outputs/cityscapes/tradeoff_matrix.pdf')
        plt.close()

    else:
        flops_per_layer = []
        rpn_layers = ['LVL_0', 'LVL_1', 'LVL_2', 'LVL_3', 'pool']
        rpn_layers_counter = 0
        detector_layers = ['FC6', 'FC7']
        detector_layers_counter = 0
        timesteps_rpn = args.num_steps_rpn
        timesteps_detector = args.num_steps_detector

        all_images_per_layer_dict = torch.load(Path(args.file))

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


if __name__ == "__main__":
    main(get_args_parser().parse_args())