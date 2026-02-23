import os
import time
import datetime
import h5py
import yaml


def get_kneadings_data(input_data_dir):
    """Gets all the results from kneadings computing stage"""
    assert input_data_dir is not None, "No input data path was given"
    assert os.path.isfile(input_data_dir), f"Data file {os.path.abspath(input_data_dir)} does not exist"

    input_data = h5py.File(input_data_dir, 'r')
    kneadings_data = input_data['kneadings_info']['kneadings_data']
    inits = input_data['sf_grid_info']['inits']
    nones = input_data['sf_grid_info']['nones']
    config = yaml.safe_load(input_data.attrs['config'])

    idxs_x, idxs_y, params_x, params_y, kneadings = kneadings_data

    idxs_x = idxs_x.astype(int)
    idxs_y = idxs_y.astype(int)

    kneadings_data = [idxs_x, idxs_y, params_x, params_y, kneadings]

    return kneadings_data, inits, nones, config


def check_config_correspondence(prev_config, curr_config, task_names):
    """Compares configs of previous and current tasks.
    Default system, grid and given tasks parameters must correspond to each other"""

    assert prev_config['defaultSystem'] == curr_config['defaultSystem'], "Default system parameters in configs differ"
    assert prev_config['grid'] == curr_config['grid'], "Grid parameters in configs differ"

    for task_name in task_names:
        prev_task_dict = prev_config[task_name].copy()
        curr_task_dict = curr_config[task_name].copy()
        prev_task_dict.pop('input_data', None)
        curr_task_dict.pop('input_data', None)
        assert prev_task_dict == curr_task_dict, f"Task {task_name} parameters in configs differ"


def general_engine(worker, configDict, startTime, initResult, dataGrid):
    start = time.time()
    workerResult = worker(config=configDict, initResult=initResult, timeStamp=startTime)
    end = time.time()
    print(f"Took {end - start}s ({datetime.timedelta(seconds=end - start)})")
    return workerResult