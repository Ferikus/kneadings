import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def set_random_color_map(system_dim, kneadings_len):
    # количество цветов без ошибок
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
    if d_first_range_max == 0:
        red[d_first] = 0.0
    else:
        red[d_first] = d_vals[d_first] / d_first_range_max

    # вторая половина отрезка -> в синий канал, красный = 0
    d_second = d_vals >= midpoint
    d_second_vals = d_vals[d_second]
    d_second_min = np.min(d_second_vals)
    d_second_max = np.max(d_second_vals)

    d_second_range_max = d_second_max - d_second_min
    if d_second_range_max == 0:
        blue[d_second] = 0.0
    else:
        blue[d_second] = (d_second_vals - d_second_min) / d_second_range_max

    # зелёный канал принимает случайные значения
    np.random.seed(7)
    green = np.random.random(color_map_levels)

    RGB = np.column_stack((red, green, blue))
    custom_cmap = ListedColormap(RGB)

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

    special_mask = grid_matrix < 0
    normal_mask = grid_matrix >= 0

    if np.any(normal_mask):
        plt.pcolormesh(unique_params_x, unique_params_y,
                       np.ma.masked_where(special_mask, grid_matrix),
                       cmap=set_color_map(),
                       shading='nearest',
                       vmin=0, vmax=1)

    if np.any(special_mask):
        min_val = -1.0
        max_val = -0.1

        negative_normalized = (grid_matrix[special_mask] - min_val) / (max_val - min_val)

        gradient_data = np.full_like(grid_matrix, np.nan)
        gradient_data[special_mask] = negative_normalized

        plt.pcolormesh(unique_params_x, unique_params_y,
                       gradient_data,
                       cmap='gray',
                       shading='nearest',
                       vmin=0, vmax=1)

    plt.xlabel(f'${param_x_caption}$')
    plt.ylabel(f'${param_y_caption}$')
    plt.tick_params(axis='x',)
    plt.tick_params(axis='y')
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)

    return fig