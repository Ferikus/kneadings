from itertools import groupby
from functools import partial

from src.plotting.convert import convert_heavy_tail_to_sequence
from src.plotting.plot_attractors import plot_attractors_plt, plot_saddle_at_sepbif
from src.routing.route_exploring import *


def get_target_points_sepbif(idxs, coords, vals):
    start_pts = []
    end_pts = []

    for val, group in groupby(enumerate(vals), key=lambda x: x[1]):
        group = list(group)
        start_idx = group[0][0]
        end_idx = group[-1][0]

        start_pts.append(make_target_point(idxs, coords, start_idx, val))
        end_pts.append(make_target_point(idxs, coords, end_idx, val))

    start_pts.pop(0)
    end_pts.pop()

    mid_pts_coords = []
    for sp, ep in zip(start_pts, end_pts):
        mid_pt_coords = tuple((sp_w + ep_w) / 2 for sp_w, ep_w in zip(sp['coords'], ep['coords']))
        mid_pts_coords.append(mid_pt_coords)

    target_pts = {
        'start_pts': start_pts,
        'end_pts': end_pts
    }

    return target_pts, mid_pts_coords


def plot_target_attractors_sepbif(config, views, saving_directory, target_pts, convert_func):
    start_pts = target_pts['start_pts']
    end_pts = target_pts['end_pts']

    default_w = float(config['defaultSystem']['w'])
    default_r = float(config['defaultSystem']['r'])

    kneadings_dict = config['kneadings_fbpo']
    dt = kneadings_dict['dt']
    n = kneadings_dict['n']
    kneadings_start = kneadings_dict['kneadings_start']
    kneadings_end = kneadings_dict['kneadings_end']

    kneadings_len = kneadings_end - kneadings_start + 1


    assert len(start_pts) == len(end_pts), "Start and end groups of points are different in length"
    for i in range(len(start_pts)):
        start_pt_a, start_pt_b = start_pts[i]['coords']
        start_pt_val = start_pts[i]['val']

        end_pt_a, end_pt_b = end_pts[i]['coords']
        end_pt_val = end_pts[i]['val']

        end_pt_val_converted = convert_func(end_pt_val, 4, kneadings_len)
        start_pt_val_converted = convert_func(start_pt_val, 4, kneadings_len)

        params1 = [default_w, end_pt_a, end_pt_b, default_r]
        params2 = [default_w, start_pt_a, start_pt_b, default_r]
        params_set = [params1, params2]

        print(f"Generating phase portrait for point {i}:\n"
              f"from ({end_pt_a:.13f}, {end_pt_b:.13f}) to ({start_pt_a:.13f}, {start_pt_b:.13f}),\n"
              f"from {end_pt_val_converted} to {start_pt_val_converted}")

        draw_saddle_wrapper = partial(plot_saddle_at_sepbif, params1=params1, params2=params2,
                                      threshold=0.15, n=30000, dt=0.01)

        plot_attractors_plt(
            params_set,
            views,
            plot_placeholder=draw_saddle_wrapper,
            start_pt=0,
            n=n,
            dt=dt,
            directory=saving_directory,
            point_name=f"sepbif_{i}_from_{end_pt_val_converted}_to_{start_pt_val_converted}"
        )


def map_out_sepbif_route_on_kneadings_set(config, kneadings_data, views):
    """Строит бифуркации сепаратрис на фазовом портрете вдоль линии среза на карте нидингов"""
    map_out_route_on_kneadings_set(config=config, kneadings_data=kneadings_data, views=views,
                                   get_target_points_func=get_target_points_sepbif,
                                   plot_target_attractors_func=plot_target_attractors_sepbif,
                                   convert_func=convert_heavy_tail_to_sequence,
                                   output_suffix="sepbif_analysis")
