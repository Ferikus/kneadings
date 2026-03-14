import os
import datetime
import matplotlib.pyplot as plt
from ast import literal_eval

from lib.computation_template.workers_utils import makeFinalOutname
from src.plotting.plot_mode_map import plot_mode_map, set_random_color_map
from src.system_analysis.thetrahedron import *
from src.computing.engines_kneadings_fbpo import (save_kneadings_data, get_kneadings_data, get_kneadings_records_data,
                                                  get_mode_map_data, get_inits_data)


def get_grid_points_along_line(data, pt1, pt2):
    """Находит узлы сетки, через которые проходит линия, используя реальные координаты."""
    idxs_x, idxs_y, params_x, params_y, vals = data

    unique_params_x = np.unique(params_x)
    unique_params_y = np.unique(params_y)

    param_x_count = len(unique_params_x)
    param_y_count = len(unique_params_y)

    grid_matrix = np.full((param_y_count, param_x_count), -0.3)
    grid_matrix[idxs_y, idxs_x] = vals

    num_points = 100  # разбиение -> вынести в аргументы функции
    t = np.linspace(0, 1, num_points)
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

    plot_settings = config['misc']['plot_settings']['default']
    accent_color = 'white'

    def set_color_map():
        return set_random_color_map(4, kneadings_len)
    plot_mode_map(kneadings_data, set_color_map, param_x_caption, param_y_caption, plot_settings)

    # отрисовка среза
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], solid_capstyle='round', c='black')

    # отрисовка точек на срезе
    for coords in rep_pts_coords:
        rep_pt_x, rep_pt_y = coords
        plt.scatter(rep_pt_x, rep_pt_y, marker='o', color=accent_color, linewidths=3, edgecolor='black', zorder=3)

    plt.title(f"(${param_x_caption}$, ${param_y_caption}$)-parameter sweep "
              f"of [{kneadings_start + 1}-{kneadings_end + 1}] length")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/map.pdf", bbox_inches='tight')
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


def map_out_route_on_kneadings_set(config, output_suffix, get_target_points_func=None,
                                   plot_target_attractors_func=None, convert_func=None):
    """General function for routing given kneading slice"""

    kneadings_input_data_path = config['kneadings']['input_data']
    kneadings_data = get_kneadings_data(kneadings_input_data_path)
    kneadings_records = get_kneadings_records_data(kneadings_input_data_path)
    mode_map_data = get_mode_map_data(kneadings_input_data_path)
    inits, nones, inner_sf_set = get_inits_data(kneadings_input_data_path)
    map_only = config['route']['map_only']

    pt1 = tuple(map(float, literal_eval(config['route']['start_pt'])))
    pt2 = tuple(map(float, literal_eval(config['route']['end_pt'])))

    output_dir = config['output']['directory']
    start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    saving_dir = os.path.join(output_dir, f"{config['output']['mask']}_{output_suffix}_{start_time}")
    os.makedirs(saving_dir, exist_ok=True)

    views = config['misc']['views']

    # сбор репрезентативных точек
    print("Getting representative points...")
    idxs, coords, vals = get_grid_points_along_line(kneadings_data, pt1, pt2)
    target_pts, rep_pts_coords = get_target_points_func(idxs, coords, vals)

    # отрисовка карты режимов
    print("Slicing the mode map...")
    slice_mode_map(config, kneadings_data, rep_pts_coords, pt1, pt2, saving_dir)

    if not map_only:
        # генерация аттракторов для каждой точки
        print("Plotting attractors for target points...")
        plot_target_attractors_func(config, views, saving_dir, target_pts, convert_func)

    # дублирование hdf5 файла в saving_directory с обновлённым значением задачи route в конфиге
    hdf5_outname = makeFinalOutname(config, {'targetDir': saving_dir}, "hdf5", start_time)
    save_kneadings_data(hdf5_outname, kneadings_data, kneadings_records, mode_map_data, inits, nones, inner_sf_set, config)
    print("Dataset successfully saved")
