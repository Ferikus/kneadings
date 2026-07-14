import os
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from ast import literal_eval

from lib.computation_template.workers_utils import makeFinalOutname
from src.plotting.plot_mode_map import plot_mode_map, set_random_color_map
from src.system_analysis.thetrahedron import *
from src.computing.engines import (save_data, get_kneadings_data, get_kneadings_records_data,
                                   get_mode_map_data, get_inits_data)


def get_grid_points_along_line(data, pt1, pt2, p):
    """Gathers grid nodes intersected by the line which is defined by two points"""
    idxs_x, idxs_y, params_x, params_y, vals = data

    unique_params_x = np.unique(params_x)
    unique_params_y = np.unique(params_y)

    param_x_count = len(unique_params_x)
    param_y_count = len(unique_params_y)

    grid_matrix = np.full((param_y_count, param_x_count), -0.3)
    grid_matrix[idxs_y, idxs_x] = vals

    t = np.linspace(0, 1, p)
    line_xs = pt1[0] + t * (pt2[0] - pt1[0])
    line_ys = pt1[1] + t * (pt2[1] - pt1[1])

    pts_idxs = []    # индексы точек
    pts_coords = []  # координаты точек
    pts_vals = []    # значение в точках

    # для каждой точки линии ищем ближайший узел сетки
    for lx, ly in zip(line_xs, line_ys):
        idx_x = np.searchsorted(unique_params_x, lx)
        idx_y = np.searchsorted(unique_params_y, ly)

        x = unique_params_x[idx_x]
        y = unique_params_y[idx_y]

        val = grid_matrix[idx_y, idx_x]

        pts_idxs.append((idx_x, idx_y))
        pts_coords.append((x, y))
        pts_vals.append(val)

    return pts_idxs, pts_coords, pts_vals


def slice_mode_map(config, kneadings_data, rep_pts_coords, pt1, pt2, save_dir):
    """Рисует карту режимов и выбранный маршрут"""
    grid_dict = config['grid']
    param_x_caption = grid_dict['first']['caption']
    param_y_caption = grid_dict['second']['caption']

    kneadings_dict = config['kneadings']
    kneadings_start = kneadings_dict['kneadings_start']
    kneadings_end = kneadings_dict['kneadings_end']
    kneadings_len = kneadings_end - kneadings_start + 1

    img_ext = config['output']['imageExtension']
    accent_color = 'white'

    plot_settings = config['misc']['plot_settings']['2d']
    plt.rcParams.update(plot_settings)

    def set_color_map():
        return set_random_color_map(4, kneadings_len)
    plot_mode_map(kneadings_data, set_color_map, param_x_caption, param_y_caption, plot_settings)

    # отрисовка среза
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], solid_capstyle='round', c='black')

    # отрисовка точек на срезе
    for coords in rep_pts_coords:
        rep_pt_x, rep_pt_y = coords
        plt.scatter(rep_pt_x, rep_pt_y, marker='o', color=accent_color, s=100, linewidths=3, edgecolor='black', zorder=3)

    plt.title(f"(${param_x_caption}$, ${param_y_caption}$)-parameter sweep "
              f"of [{kneadings_start + 1}-{kneadings_end + 1}] length")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/map.{img_ext}", bbox_inches='tight')
    plt.show()


def make_target_point(idxs_list, coords_list, idx, val):
    """Makes a target point on the mode map"""
    rep_pt_idx = idxs_list[idx]
    rep_pt_coords = coords_list[idx]

    return {
        'idx': rep_pt_idx,
        'coords': rep_pt_coords,
        'val': val
    }