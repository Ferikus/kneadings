from itertools import groupby

from src.plotting.convert import convert_heavy_tail_to_sequence
from src.plotting.plot_attractors import plot_attractors_plt
from src.routing.route_exploring import *


def get_target_points_attr(idxs, coords, vals):
    rep_pts = []

    for val, group in groupby(enumerate(vals), key=lambda x: x[1]):
        if val < 0:
            continue

        group_indices = [i for i, _ in group]
        mid_idx = group_indices[len(group_indices) // 2]

        rep_pts.append(make_target_point(idxs, coords, mid_idx, val))

    slice_coords = [pt['coords'] for pt in rep_pts]

    return rep_pts, slice_coords


def plot_target_attractors_attr(config, views, saving_directory, plotting_data, convert_func):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']
    param_to_index = def_sys_dict['param_to_index']

    grid_dict = config['grid']
    param_x_name = grid_dict['first']['name']
    param_y_name = grid_dict['second']['name']

    kneadings_dict = config['kneadings']
    dt = kneadings_dict['dt']
    n = kneadings_dict['n']
    kneadings_start = kneadings_dict['kneadings_start']
    kneadings_end = kneadings_dict['kneadings_end']

    params = [w, a, b, r]
    kneadings_len = kneadings_end - kneadings_start + 1

    print("Generating phase portraits...")
    for i, rep_pt in enumerate(plotting_data):
        param_x, param_y = rep_pt['coords']
        val = rep_pt['val']

        val_converted = convert_func(val, 4, kneadings_len)

        params[param_to_index[param_x_name]] = param_x
        params[param_to_index[param_y_name]] = param_y
        params_set = [params]

        print(f"Generating phase portrait for point {i}: ({param_x:.13f}, {param_y:.13f}) at sequence {val_converted}")

        plot_attractors_plt(
            params_set,
            views=views,
            plot_placeholder=None,
            start_pt=0,
            n=n,
            dt=dt,
            directory=saving_directory,
            point_name=f"attr_{i}_{val_converted}"
        )


def map_out_attr_route_on_kneadings_set(config):
    """Строит бифуркации сепаратрис на фазовом портрете вдоль линии среза на карте нидингов"""
    map_out_route_on_kneadings_set(config=config, output_suffix="attr_analysis",
                                   get_target_points_func=get_target_points_attr,
                                   plot_target_attractors_func=plot_target_attractors_attr,
                                   convert_func=convert_heavy_tail_to_sequence,)
