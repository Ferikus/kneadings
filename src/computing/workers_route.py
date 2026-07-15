import os
from ast import literal_eval

from src.computing.engines import (save_data, get_data, get_records_data,
                                   get_mode_map_data, get_inits_data, get_config_data)
from src.routing.route_exploring import get_grid_points_along_line, slice_mode_map
from src.routing.route_tools_attr import get_target_points_attr, plot_target_attractors_attr
from src.routing.route_tools_sepbif import get_target_points_sepbif, plot_target_attractors_sepbif
from src.system_analysis.convert import convert_heavy_tail_to_sequence

### to connect with workers file
from lib.computation_template.workers_utils import register, makeFinalOutname
from src.computing.workers import registry


@register(registry, 'init', 'route')
def init_route(config, timeStamp):
    kneadings_input_data_path = config['kneadings']['input_data']
    kneadings_config = get_config_data(kneadings_input_data_path)
    kneadings_data = get_data(kneadings_input_data_path, kneadings_config)
    kneadings_records = get_records_data(kneadings_input_data_path, kneadings_config)
    mode_map_data = get_mode_map_data(kneadings_input_data_path, kneadings_config)
    inits, nones, inner_sf_set = get_inits_data(kneadings_input_data_path)

    pt1 = tuple(map(float, literal_eval(config['route']['start_pt'])))
    pt2 = tuple(map(float, literal_eval(config['route']['end_pt'])))

    selected = config['route']['mode']
    if selected == "attr":
        get_target_points_func = get_target_points_attr
    elif selected == "sepbif":
        get_target_points_func = get_target_points_sepbif
    else:
        raise ValueError("No such option for a route mode")

    print("Getting representative points...")
    idxs, coords, vals = get_grid_points_along_line(kneadings_data, pt1, pt2, 100)
    target_pts, rep_pts_coords = get_target_points_func(idxs, coords, vals)

    return {
        'kneadings_data': kneadings_data,
        'kneadings_records': kneadings_records,
        'mode_map_data': mode_map_data,
        'inits': inits,
        'nones': nones,
        'inner_sf_set': inner_sf_set,
        'pt1': pt1,
        'pt2': pt2,
        'target_pts': target_pts,
        'rep_pts_coords': rep_pts_coords,
        'targetDir': 'output'
    }


@register(registry, 'worker', 'route')
def worker_route(config, initResult, timeStamp):
    output_dir = config['output']['directory']
    selected = config['route']['mode']
    output_suffix = f"{selected}_analysis"
    saving_dir = os.path.join(output_dir, f"{config['output']['mask']}_{output_suffix}_{timeStamp}")
    os.makedirs(saving_dir, exist_ok=True)

    map_only = config['route']['map_only']
    if not map_only:
        views = config['misc']['views']
        target_pts = initResult['target_pts']

        if selected == "attr":
            plot_target_attractors_func = plot_target_attractors_attr
        elif selected == "sepbif":
            plot_target_attractors_func = plot_target_attractors_sepbif

        print("Plotting attractors for target points...")
        plot_target_attractors_func(config, views, saving_dir, target_pts, convert_heavy_tail_to_sequence)

    return {'saving_dir': saving_dir}


@register(registry, 'post', 'route')
def post_route(config, initResult, workerResult, grid, startTime):
    kneadings_data = initResult['kneadings_data']
    kneadings_records = initResult['kneadings_records']
    mode_map_data = initResult['mode_map_data']
    inits = initResult['inits']
    nones = initResult['nones']
    inner_sf_set = initResult['inner_sf_set']
    pt1 = initResult['pt1']
    pt2 = initResult['pt2']
    rep_pts_coords = initResult['rep_pts_coords']

    saving_dir = workerResult['saving_dir']

    print("Slicing the mode map...")
    slice_mode_map(config, kneadings_data, rep_pts_coords, pt1, pt2, saving_dir)

    hdf5_outname = makeFinalOutname(config, {'targetDir': saving_dir}, "hdf5", startTime)
    save_data(hdf5_outname, kneadings_data, kneadings_records, mode_map_data, inits, nones, inner_sf_set, config)
    print("Dataset successfully saved")
