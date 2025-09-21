import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.mapping.normalization import *
from src.mapping.convert import decimal_to_binary, binary_to_decimal


def set_random_color_map():
    colorMapLevels = 2**8
    blue = np.linspace(0.01, 1.0, colorMapLevels)
    red = 1 - blue
    green = np.random.random(colorMapLevels) * 0.8
    # green = np.linspace(0.8, 1.0, colorMapLevels)
    RGB = np.column_stack((red, green, blue))
    custom_cmap = ListedColormap(RGB)
    return custom_cmap

def set_gradient_color_map():
    colorMapLevels = 2**8
    blue = np.linspace(0.01, 1.0, colorMapLevels)
    red = 1 - blue
    green = np.linspace(0.8, 1.0, colorMapLevels)
    RGB = np.column_stack((red, green, blue))
    custom_cmap = ListedColormap(RGB)
    return custom_cmap


def plot_mode_map(kneadings_weighted_sum_set, set_color_map, param_x_caption, param_y_caption,
                  param_x_start, param_x_end, param_x_count, param_y_start, param_y_end, param_y_count,
                  font_size):
    custom_cmap = set_color_map()

    # kneadings_norm = []
    # for i in range(len(kneadings_weighted_sum_set)):
    #     if kneadings_weighted_sum_set[i] in [-0.1, -0.2]:
    #         kneadings_norm.append(kneadings_weighted_sum_set[i])
    #         continue
    #     kneading_bin = decimal_to_binary(kneadings_weighted_sum_set[i])
    #     kneading_bin_norm = normalize_kneading(kneading_bin)
    #     kneading_dec_norm = binary_to_decimal(kneading_bin_norm[0])
    #     print(f"{kneadings_weighted_sum_set[i]} ({kneading_bin}) после нормализации: {kneading_dec_norm} ({kneading_bin_norm})")
    #     kneadings_norm.append(kneading_dec_norm)
    #     # kneadings_norm.append(1.0/len(kneading_bin_norm[1]))

    plt.figure(figsize=(8, 8))
    plt.imshow(
        np.reshape(kneadings_weighted_sum_set, (param_x_count, param_y_count), 'F'),
        extent=[param_x_start, param_x_end, param_y_start, param_y_end],
        cmap=custom_cmap,
        vmin=-0.3,
        vmax=1,
        origin='lower',
        aspect='auto'
    )
    plt.xlabel(f'${param_x_caption}$', fontsize=font_size)
    plt.ylabel(f'${param_y_caption}$', fontsize=font_size)
    plt.tick_params(axis='x', labelsize=font_size)
    plt.tick_params(axis='y', labelsize=font_size)


if __name__ == "__main__":
    # data = np.load(r'../kneadings_results.npz')
    # kneadings_weighted_sum_set = data['results']
    # sweep_size = data['sweep_size']
    # a_start = data['a_start']
    # a_end = data['a_end']
    # b_start = data['b_start']
    # b_end = data['b_end']

    data_kneadings = np.load(r'../cuda_sweep/sweep_fbpo.npz')
    data_inits = np.load(r'../system_analysis/inits.npz')

    kneadings_weighted_sum_set = data_kneadings['kneadings']
    param_x_count = data_inits['left_n'] + data_inits['right_n'] + 1
    param_y_count = data_inits['up_n'] + data_inits['down_n'] + 1
    param_x_start = -2.67 - data_inits['left_n'] * 0.01
    param_x_end = -2.67 + data_inits['right_n'] * 0.01
    param_y_start = -1.61268422884276 - data_inits['down_n'] * 0.01
    param_y_end = -1.61268422884276 + data_inits['up_n'] * 0.01

    plot_mode_map(kneadings_weighted_sum_set, set_random_color_map, r'\alpha', r'\beta',
                  param_x_start, param_x_end, param_x_count, param_y_start, param_y_end, param_y_count, 12)
    plt.title(r'$\omega = 0$, $r = 1$', fontsize=12)
    plt.savefig("mode_map_1.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Mode map successfully saved")