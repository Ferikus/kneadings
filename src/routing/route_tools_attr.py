from itertools import groupby

from src.plotting.convert import convert_heavy_tail_to_sequence
from src.plotting.plot_attractors import plot_attractors_plt
from src.routing.route_exploring import *


def get_target_points_attr(idxs, coords, vals):
    rep_pts = []

    for val, group in groupby(enumerate(vals), key=lambda x: x[1]):
        if val in [-0.1, -0.2, -0.3]:
            continue

        group_indices = [i for i, _ in group]
        mid_idx = group_indices[len(group_indices) // 2]

        rep_pts.append(make_target_point(idxs, coords, mid_idx, val))

    slice_coords = [pt['coords'] for pt in rep_pts]

    return rep_pts, slice_coords


def plot_target_attractors_attr(config, views, saving_directory, plotting_data, convert_func):
    default_w = float(config['defaultSystem']['w'])
    default_r = float(config['defaultSystem']['r'])

    kneadings_dict = config['kneadings']
    kneadings_start = kneadings_dict['kneadings_start']
    kneadings_end = kneadings_dict['kneadings_end']

    kneadings_len = kneadings_end - kneadings_start + 1

    print("Generating phase portraits...")

    for i, rep_pt in enumerate(plotting_data):
        a, b = rep_pt['coords']
        val = rep_pt['val']

        val_converted = convert_func(val, 4, kneadings_len)

        params = [default_w, a, b, default_r]
        params_set = [params]

        print(f"Generating phase portrait for point {i}: ({a:.13f}, {b:.13f}) at sequence {val_converted}")

        plot_attractors_plt(
            params_set,
            views=views,
            plot_placeholder=None,
            start_pt=0,
            n=50000,
            dt=0.01,
            directory=saving_directory,
            point_name=f"attr_{i}_{val_converted}"
        )


def map_out_attr_route_on_kneadings_set(config):
    """Строит бифуркации сепаратрис на фазовом портрете вдоль линии среза на карте нидингов"""
    map_out_route_on_kneadings_set(config=config, output_suffix="attr_analysis",
                                   get_target_points_func=get_target_points_attr,
                                   plot_target_attractors_func=plot_target_attractors_attr,
                                   convert_func=convert_heavy_tail_to_sequence,)
