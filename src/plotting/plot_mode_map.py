import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def set_random_color_map(system_dim, kneadings_len):
    color_map_levels = system_dim ** kneadings_len

    # d принимает целые значения на отрезке [0; 2**q - 1]
    # дальше идёт обработка каналов до и после середины интервала
    d_vals = np.arange(color_map_levels)
    midpoint = (color_map_levels - 1) / 2.0

    red = np.zeros(color_map_levels)
    blue = np.zeros(color_map_levels)

    # первый полуинтервал отрезка -> в красный канал, синий = 0
    d_first = d_vals < midpoint
    d_first_range_max = np.max(d_vals[d_first])
    red[d_first] = d_vals[d_first] / d_first_range_max

    # вторая половина отрезка -> в синий канал, красный = 0
    d_second = d_vals >= midpoint
    d_second_vals = d_vals[d_second]
    d_second_min = np.min(d_second_vals)
    d_second_max = np.max(d_second_vals)

    d_second_range_max = d_second_max - d_second_min
    blue[d_second] = (d_second_vals - d_second_min) / d_second_range_max

    # зелёный канал принимает случайные значения
    np.random.seed(7)
    green = np.random.random(color_map_levels)

    RGB = np.column_stack((red, green, blue))
    custom_cmap = ListedColormap(RGB)

    # обработка особых цветов для ошибок
    special_colors = {
        -0.3: [0, 0, 0],            # NoInitFoundError  черный
        -0.20: [0.25, 0.25, 0.25],  # InfinityError  тёмно-серый
        -0.21: [0.9, 0.9, 0.9],  # InEquilibriumError светло-серый
        -0.1: [1, 1, 1]             # KneadingDoNotEndError  белый
    }

    total_range = 1 - (-0.3)
    for value, color in special_colors.items():
        idx = int((value - (-0.3)) / total_range * (color_map_levels - 1))
        custom_cmap.colors[idx] = color

    return custom_cmap


def set_mode_map_size(param_x_count, param_y_count):
    size_x = param_x_count
    size_y = param_y_count
    max_size = 10.

    if size_x > size_y:
        size_y *= max_size / size_x
        size_x = max_size
    else:
        size_x *= max_size / size_y
        size_y = max_size

    return size_x, size_y


def plot_mode_map(kneadings_data, set_color_map, param_x_caption, param_y_caption, plot_settings):
    """Строит карту режимов на основе таблицы данных нидингов"""
    mpl.rcParams.update(plot_settings)

    idxs_x, idxs_y, params_x, params_y, kneadings = kneadings_data

    unique_params_x = np.unique(params_x)
    unique_params_y = np.unique(params_y)

    param_x_count = len(unique_params_x)
    param_y_count = len(unique_params_y)

    grid_matrix = np.full((param_y_count, param_x_count), -0.3)  # строки x столбцы
    grid_matrix[idxs_y, idxs_x] = kneadings

    fig = plt.figure(figsize=set_mode_map_size(param_x_count, param_y_count))

    plt.pcolormesh(unique_params_x, unique_params_y, grid_matrix,
                   cmap=set_color_map(),
                   shading='nearest',  # центрует ячейку на координате (x, y)
                   vmin=-0.3,
                   vmax=1)

    plt.xlabel(f'${param_x_caption}$')
    plt.ylabel(f'${param_y_caption}$')
    plt.tick_params(axis='x',)
    plt.tick_params(axis='y')
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)

    return fig