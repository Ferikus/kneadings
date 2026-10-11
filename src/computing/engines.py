import os
import time
import datetime
import h5py
import yaml
import numpy as np
import matplotlib.pyplot as plt

from src.plotting.plot_mode_map import plot_mode_map, make_set_color_map

TASK_COMPATIBILITY_GROUPS = [
    {'regularity', 'periodicity', 'complexity'},
]


def is_compatible_task(curr_task, source_task):
    if curr_task == source_task:
        return True
    for group in TASK_COMPATIBILITY_GROUPS:
        if curr_task in group and source_task in group:
            return True
    return False


def get_data(input_data_dir, config):
    """Gets task data from task computing stage"""
    assert input_data_dir is not None, "No input data path was given"
    assert os.path.isfile(input_data_dir), f"Data file {os.path.abspath(input_data_dir)} does not exist"

    curr_task = config['task']
    input_data = h5py.File(input_data_dir, 'r')
    source_task = None
    for key in input_data.keys():
        if key.endswith('_info') and key != 'sf_grid_info':
            source_task = key[:-5]
            break

    assert is_compatible_task(curr_task, source_task), \
        f"Task '{curr_task}' cannot import data produced by '{source_task}'"

    data = input_data[f'{source_task}_info'][f'{source_task}_data']

    idxs_x, idxs_y, params_x, params_y, vals = data

    idxs_x = idxs_x.astype(int)
    idxs_y = idxs_y.astype(int)

    data = [idxs_x, idxs_y, params_x, params_y, vals]

    return data


def get_mode_map_data(input_data_dir, config):
    assert input_data_dir is not None, "No input data path was given"
    assert os.path.isfile(input_data_dir), f"Data file {os.path.abspath(input_data_dir)} does not exist"

    input_data = h5py.File(input_data_dir, 'r')
    task_name = config['task']
    mode_map_data = input_data[f'{task_name}_info']['mode_map_data']

    return mode_map_data


def get_inits_data(input_data_dir):
    assert input_data_dir is not None, "No input data path was given"
    assert os.path.isfile(input_data_dir), f"Data file {os.path.abspath(input_data_dir)} does not exist"

    input_data = h5py.File(input_data_dir, 'r')
    inits = list(input_data['sf_grid_info']['inits'])
    nones = list(input_data['sf_grid_info']['nones'])
    inner_sf_set = list(input_data['sf_grid_info']['inner_sf_set'])

    return inits, nones, inner_sf_set


def get_records_data(input_data_dir, config):
    assert input_data_dir is not None, "No input data path was given"
    assert os.path.isfile(input_data_dir), f"Data file {os.path.abspath(input_data_dir)} does not exist"

    input_data = h5py.File(input_data_dir, 'r')
    task_name = config['task']
    records = input_data[f'{task_name}_info'][f'{task_name}_records'][()].decode('utf-8')

    return records


def get_config_data(input_data_dir):
    assert input_data_dir is not None, "No input data path was given"
    assert os.path.isfile(input_data_dir), f"Data file {os.path.abspath(input_data_dir)} does not exist"

    input_data = h5py.File(input_data_dir, 'r')
    config = yaml.safe_load(input_data.attrs['config'])

    return config


def save_data(h5py_outname, data, records, mode_map_data, inits, nones, inner_sf_set, config):
    """Saves all the results from computing stage"""
    task_name = config['task']
    with h5py.File(h5py_outname, 'w') as main_folder:
        data_info = main_folder.create_group(f'{task_name}_info')
        data_info.create_dataset(f'{task_name}_data', data=data)
        data_info.create_dataset(f'{task_name}_records', data=records)
        data_info.create_dataset('mode_map_data', data=mode_map_data)

        sf_grid_info = main_folder.create_group('sf_grid_info')
        sf_grid_info.create_dataset('inits', data=inits)
        sf_grid_info.create_dataset('nones', data=nones)
        sf_grid_info.create_dataset('inner_sf_set', data=inner_sf_set)

        config_string = yaml.dump(config)
        main_folder.attrs['config'] = config_string


def check_config_correspondence(source_config, curr_config, task_names, check_source=False):
    """Compares configs of previous and current tasks.
    Default system, grid and given tasks parameters must correspond to each other"""
    print("Checking config correspondence...")

    assert source_config['defaultSystem'] == curr_config['defaultSystem'],\
        f"Default system parameters in configs differ\n{yaml.dump(source_config['defaultSystem'])}"
    assert source_config['grid'] == curr_config['grid'],\
        f"Grid parameters in configs differ\n{yaml.dump(source_config['grid'])}"

    if check_source:
        source_objective_dict = source_config[source_config['task']].copy()
        curr_objective_dict = curr_config[curr_config['task']].copy()
        source_objective_dict.pop('input_data', None)
        curr_objective_dict.pop('input_data', None)
        assert source_objective_dict == curr_objective_dict, \
            (f"Current task '{curr_config['task']}' parameters in configs differ with source task '{source_config['task']}'"
             f"\n{yaml.dump(source_objective_dict)}")

    # task_names -- другие задачи, кроме сорс
    for task_name in task_names:
        source_task_dict = source_config[task_name].copy()
        curr_task_dict = curr_config[task_name].copy()
        source_task_dict.pop('input_data', None)
        curr_task_dict.pop('input_data', None)
        assert source_task_dict == curr_task_dict,\
            f"Task {task_name} parameters in configs differ\n{yaml.dump(source_config[task_name])}"


def get_params_from_config(data_path):
    config = get_config_data(data_path)
    sys_dict = config['defaultSystem']
    params = [sys_dict['w'], sys_dict['a'], sys_dict['b'], sys_dict['r']]
    return params


def plot_mode_map_by_config(data_path):
    kneadings_data = get_data(data_path)
    config = get_config_data(data_path)
    param_x_caption = config['grid']['first']['caption']
    param_y_caption = config['grid']['second']['caption']
    plot_settings = config['misc']['plot_settings']['default']
    kneadings_start = config['kneadings']['kneadings_start']
    kneadings_end = config['kneadings']['kneadings_end']
    kneadings_len = kneadings_end - kneadings_start + 1
    set_color_map = make_set_color_map(kneadings_len)
    fig = plot_mode_map(kneadings_data, set_color_map, param_x_caption, param_y_caption, plot_settings)
    plt.title(f"(${param_x_caption}$, ${param_y_caption}$)-parameter sweep of [{kneadings_start + 1}-{kneadings_end + 1}] length")
    return fig


def general_engine(worker, configDict, startTime, initResult, dataGrid):
    start = time.time()
    workerResult = worker(config=configDict, initResult=initResult, timeStamp=startTime)
    end = time.time()
    print(f"Took {end - start}s ({datetime.timedelta(seconds=end - start)})")
    return workerResult