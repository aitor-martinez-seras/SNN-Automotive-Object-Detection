import json

import numpy as np
import matplotlib.pyplot as plt


def main():

    with open(r"outputs/cityscapes/noise_acc_model_Cityscapes_SNN.json", 'r') as fp:
        noise_snn = json.load(fp)
    with open(r"outputs/cityscapes/noise_acc_model_Cityscapes_SNN_FPN_tuned.json", 'r') as fp:
        noise_snn_tuned = json.load(fp)
    with open(r"outputs/cityscapes/noise_acc_model_Cityscapes_NoSNN.json", 'r') as fp:
        noise_nosnn = json.load(fp)
    with open(r"outputs/cityscapes/noise_acc_model_Cityscapes_NoSNN_FPN_tuned.json", 'r') as fp:
        noise_nosnn_tuned = json.load(fp)

    # Extract noises
    metric = 3
    std_dev_values = np.array([x[1] for x in noise_snn[:25]])

    snn_base_perf = noise_snn[0][metric]
    snn_tuned_base_perf = noise_snn_tuned[0][metric]
    nosnn_base_perf = noise_nosnn[0][metric]
    nosnn_tuned_base_perf = noise_nosnn_tuned[0][metric]

    gaussian_accuracy_drop_snn = np.array([(snn_base_perf - x[metric]) / snn_base_perf for x in noise_snn[:25]])
    gaussian_accuracy_drop_snn_tuned = np.array([(snn_tuned_base_perf - x[metric]) / snn_tuned_base_perf for x in noise_snn_tuned[:25]])
    gaussian_accuracy_drop_nosnn = np.array([(nosnn_base_perf - x[metric]) / nosnn_base_perf for x in noise_nosnn[:25]])
    gaussian_accuracy_drop_nosnn_tuned = np.array([(nosnn_tuned_base_perf - x[metric]) / nosnn_tuned_base_perf for x in noise_nosnn_tuned[:25]])

    # Define the params for the plot
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

    x_size, y_size = 3.5, 3.5
    fig, ax = plt.subplots(1, 1, figsize=(x_size, y_size))
    ax.plot(std_dev_values[:25], gaussian_accuracy_drop_snn, label=r'$\textit{SNN}$')
    ax.plot(std_dev_values[:25], gaussian_accuracy_drop_snn_tuned, label=r'$\textit{SNN}_{*}$')
    ax.plot(std_dev_values[:25], gaussian_accuracy_drop_nosnn, label=r'$\textit{NoSNN}$')
    ax.plot(std_dev_values[:25], gaussian_accuracy_drop_nosnn_tuned, label=r'$\textit{NoSNN}_{*}$')
    ax.set_ylabel('Precision drop')
    ax.set_xlabel('Noise standard deviation')
    ax.legend(loc='lower right')
    fig.savefig('outputs/cityscapes/gaussian.pdf', bbox_inches='tight')
    plt.close()

    with open(r"outputs/cityscapes/rain_noise_acc_model_Cityscapes_SNN.json", 'r') as fp:
        noise_snn = json.load(fp)
    with open(r"outputs/cityscapes/rain_noise_acc_model_Cityscapes_SNN_FPN_tuned.json", 'r') as fp:
        noise_snn_tuned = json.load(fp)
    with open(r"outputs/cityscapes/rain_noise_acc_model_Cityscapes_NoSNN.json", 'r') as fp:
        noise_nosnn = json.load(fp)
    with open(r"outputs/cityscapes/rain_noise_acc_model_Cityscapes_NoSNN_FPN_tuned.json", 'r') as fp:
        noise_nosnn_tuned = json.load(fp)

    # Rain
    raindrop_values = np.array([x[0] for x in noise_snn])

    rain_accuracy_drop_snn = np.array([(snn_base_perf - x[metric]) / snn_base_perf for x in noise_snn])

    rain_accuracy_drop_snn_tuned = np.array([(snn_tuned_base_perf - x[metric]) / snn_tuned_base_perf for x in noise_snn_tuned])
    
    rain_accuracy_drop_nosnn = np.array([(nosnn_base_perf - x[metric]) / nosnn_base_perf for x in noise_nosnn])

    rain_accuracy_drop_nosnn_tuned = np.array([(nosnn_tuned_base_perf - x[metric]) / nosnn_tuned_base_perf for x in noise_nosnn_tuned])

    x_size, y_size = 3.5, 3.5
    fig, ax = plt.subplots(1, 1, figsize=(x_size, y_size))
    ax.plot(raindrop_values, rain_accuracy_drop_snn, label=r'$\textit{SNN}$')
    ax.plot(raindrop_values, rain_accuracy_drop_snn_tuned, label=r'$\textit{SNN}_{*}$')
    ax.plot(raindrop_values, rain_accuracy_drop_nosnn, label=r'$\textit{NoSNN}$')
    ax.plot(raindrop_values, rain_accuracy_drop_nosnn_tuned, label=r'$\textit{NoSNN}_{*}$')
    ax.set_ylabel('Precision drop')
    ax.set_xlabel('Number of raindrops')
    ax.legend(loc='lower right')
    fig.savefig('outputs/cityscapes/rain.pdf', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()