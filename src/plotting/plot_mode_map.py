import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from src.symmetry.detectives import DATA_TO_COLOR


def set_random_color_map(system_dim, kneadings_len):

    # простой способ, но не гарантирует контрастных цветов
    # color_map_levels = system_dim ** kneadings_len
    # blue = np.linspace(0.01, 1, color_map_levels)
    # red = 1 - blue
    # green = np.random.random(color_map_levels) * 0.8 + 0.1
    # RGB = np.column_stack((red, green, blue))
    # custom_cmap = colors.ListedColormap(RGB)
    # return custom_cmap

    # второй способ R-R-B-B
    # количество цветов без ошибок
    color_map_levels = system_dim ** kneadings_len

    # d принимает целые значения на отрезке [0; 2**q - 1]
    # дальше идёт обработка каналов до и после середины интервала
    d_vals = np.arange(color_map_levels, dtype=np.int32)
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
    green = np.random.random(color_map_levels) * 0.6 + 0.2

    RGB = np.column_stack((red, green, blue))
    custom_cmap = colors.ListedColormap(RGB)

    return custom_cmap


def make_set_color_map(kneadings_len):
    def set_color_map():
        return set_random_color_map(4, kneadings_len)
    return set_color_map


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


def prepare_mode_map(params_x, params_y, param_x_caption, param_y_caption, plot_settings):
    """Настройка сетки параметров и оформления осей."""
    mpl.rcParams.update(plot_settings)

    unique_params_x = np.unique(params_x)
    unique_params_y = np.unique(params_y)
    param_x_count = len(unique_params_x)
    param_y_count = len(unique_params_y)

    # fig_size = set_mode_map_size(param_x_count, param_y_count)
    # fig = plt.figure(figsize=fig_size)
    fig = plt.figure()

    plt.xlabel(f'${param_x_caption}$')
    plt.ylabel(f'${param_y_caption}$')
    # plt.xlim(unique_params_x.min(), unique_params_x.max())
    # plt.ylim(unique_params_y.min(), unique_params_y.max())

    plt.tick_params(axis='x')
    plt.tick_params(axis='y')
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)

    plt.gca().set_aspect('equal')

    return fig, unique_params_x, unique_params_y, param_x_count, param_y_count


def plot_mode_map(kneadings_data, set_color_map, param_x_caption, param_y_caption, plot_settings):
    """Строит карту режимов на основе таблицы данных нидингов"""
    idxs_x, idxs_y, params_x, params_y, kneadings = kneadings_data

    fig, unique_params_x, unique_params_y, param_x_count, param_y_count = prepare_mode_map(
        params_x, params_y, param_x_caption, param_y_caption, plot_settings
    )

    grid_matrix = np.full((param_y_count, param_x_count), -0.3)
    grid_matrix[idxs_y, idxs_x] = kneadings

    special_mask = grid_matrix < 0
    normal_mask = grid_matrix >= 0

    if np.any(normal_mask):
        plt.pcolormesh(unique_params_x, unique_params_y,
                       np.ma.masked_where(special_mask, grid_matrix),
                       cmap=set_color_map(),
                       shading='nearest',
                       vmin=0, vmax=1,
                       rasterized=True)

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
                       vmin=0, vmax=1,
                       rasterized=True)

    return fig


def plot_complexity_data(complexity_data, param_x_caption, param_y_caption, plot_settings):
    idxs_x, idxs_y, params_x, params_y, complexity_set = complexity_data

    fig, unique_x, unique_y, nx, ny = prepare_mode_map(
        params_x, params_y, param_x_caption, param_y_caption, plot_settings
    )

    grid_matrix = np.full((ny, nx), -1.0)
    grid_matrix[idxs_y, idxs_x] = complexity_set
    plot_data = np.ma.masked_where(grid_matrix < 0, grid_matrix)

    custom_cmap = plt.get_cmap('gist_rainbow').copy()
    custom_cmap.set_under('black')
    custom_cmap.set_bad('gray')

    plt.pcolormesh(unique_x, unique_y,
                   plot_data,
                   cmap=custom_cmap,
                   shading='nearest',
                   vmin=1.0, vmax=4.0,
                   rasterized=True)

    # legend
    legend_vals = np.unique([val for val in complexity_set if val > 0]).astype(int)
    norm = colors.Normalize(vmin=1.0, vmax=4.0)

    legend_patches = [
        mpatches.Patch(color='black', label='Irregular'),
        mpatches.Patch(color='gray', label='Error'),
        *[mpatches.Patch(color=custom_cmap(norm(p)), label=f'{p}') for p in legend_vals]
    ]
    plt.legend(
        handles=legend_patches,
        title="Complexity",
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.
    )

    return fig


def plot_regularity_data(regularity_data, param_x_caption, param_y_caption, plot_settings):
    idxs_x, idxs_y, params_x, params_y, regularity_set = regularity_data

    fig, unique_x, unique_y, nx, ny = prepare_mode_map(
        params_x, params_y, param_x_caption, param_y_caption, plot_settings
    )

    grid_matrix = np.full((ny, nx), -1.0)
    grid_matrix[idxs_y, idxs_x] = regularity_set
    plot_data = np.ma.masked_where(grid_matrix < 0, grid_matrix)

    custom_cmap = colors.ListedColormap(['black'])
    custom_cmap.set_over('white')
    custom_cmap.set_bad('gray')

    plt.pcolormesh(unique_x, unique_y,
                   plot_data,
                   cmap=custom_cmap,
                   shading='nearest',
                   vmin=-1e-6, vmax=1e-6,
                   rasterized=True)

    # legend
    legend_patches = [
        mpatches.Patch(color='white', label='Regular'),
        mpatches.Patch(color='black', label='Irregular'),
        mpatches.Patch(color='gray', label='Error'),
    ]
    plt.legend(
        handles=legend_patches,
        title="Regularity",
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.
    )

    return fig


def plot_detectives_data(detectives_data, proximitiesRule, param_x_caption, param_y_caption, plot_settings):
    idxs_x, idxs_y, params_x, params_y, detectives_outputs = detectives_data

    fig, unique_params_x, unique_params_y, param_x_count, param_y_count = prepare_mode_map(
        params_x, params_y, param_x_caption, param_y_caption, plot_settings
    )

    detectives_arrays = np.reshape(detectives_outputs, (-1, 4))
    # classify detectives_outputs
    symmTypes = []
    for do in detectives_arrays:
        err, *dists = do
        if err != 1.0:
            symmTypes.append(err)
        else:
            symmTypes.append(proximitiesRule(dists))

    symmTypeColored = [colors.to_rgb(DATA_TO_COLOR[st]) for st in symmTypes]

    symmColors = np.array(symmTypeColored)
    symmGrid = np.reshape(symmColors, (param_y_count, param_x_count, 3))

    plt.pcolormesh(unique_params_x, unique_params_y, symmGrid)

    # legend
    legend_patches = [
        # mpatches.Patch(color='white', label='No init error'),
        # mpatches.Patch(color='darkgray', label='Equilibrium error'),
        mpatches.Patch(color='red', label='T1'),
        mpatches.Patch(color='yellow', label='T2'),
        mpatches.Patch(color='green', label='T0'),
    ]
    plt.legend(
        handles=legend_patches,
        title="Symmetry types",
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.
    )

    return fig