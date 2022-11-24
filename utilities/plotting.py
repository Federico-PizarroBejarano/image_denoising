'''Plotting utility. '''
import pickle

import numpy as np
import matplotlib.pyplot as plt


def plot_graph(data, data_name, unit=''):
    '''Plots a graph of the given data.

    Args:
        data (dict): The dictionary containing the data.
        data_name (str): The name of the data being plotted.
        unit (str): The unit of the data (if any).
    '''

    mean_results = {'quad':0, 'TV':0, 'nlm':0, 'wnnm':0}
    labels = ['Quadratic', 'Total Variation', 'Non-Local Means', 'Weighted NNM']

    for algo in mean_results.keys():
        for image in data[algo].keys():
            mean_results[algo] += sum(data[algo][image])/15

    _, ax = plt.subplots()
    bars = ax.bar(range(1, 5), mean_results.values(), color=['cornflowerblue', 'goldenrod', 'pink', 'tomato'])
    ax.set_ylabel(f'{data_name} {unit}', fontsize=20)
    ax.set_xticks(range(1, 5), labels=labels, fontsize=20)
    ax.bar_label(bars, labels=np.round(list(mean_results.values()), 2), fontsize=15)
    ax.set_title(f'Mean {data_name} for each Denoising Algorithm', fontsize=25)
    plt.show()


if __name__ == '__main__':
    with open('./results/PSNR_results.pkl', 'rb') as f:
        PSNR_data = pickle.load(f)
    with open('./results/time_results.pkl', 'rb') as f:
        time_data = pickle.load(f)

    plot_graph(PSNR_data, 'PSNR', '')
    plot_graph(time_data, 'Runtime', '(s)')
