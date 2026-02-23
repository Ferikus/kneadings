import numpy as np
import h5py
import time
import datetime
import matplotlib.pyplot as plt
import yaml
import io

from lib.computation_template.workers_utils import register, makeFinalOutname
import lib.eq_finder.systems_fun as sf
import lib.eq_finder.SystOsscills as so

from src.computing.engines_kneadings_fbpo import get_kneadings_data, check_config_correspondence
from src.system_analysis.get_inits import (continue_equilibrium, continue_equilibrium_mp,
                                           get_eq_type_grid, find_inits_for_equilibrium_grid, generate_parameters)
from src.plotting.convert import convert_heavy_tail_to_sequence
from src.cuda_sweep.sweep_fbpo import sweep
from src.plotting.plot_mode_map import plot_mode_map, set_random_color_map

registry = {
    "worker": {},
    "init": {},
    "post": {}
}


@register(registry, 'init', 'kneadings_fbpo')
def init_kneadings_fbpo(config, timeStamp):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']
    param_to_index = def_sys_dict['param_to_index']

    sf_grid_dict = config['sf_grid']
    start_eq = sf_grid_dict['start_eq']
    input_data_path = sf_grid_dict['input_data']

    grid_dict = config['grid']
    up_n = int(grid_dict['second']['up_n'])
    up_step = float(grid_dict['second']['up_step'])
    down_n = int(grid_dict['second']['down_n'])
    down_step = float(grid_dict['second']['down_step'])
    left_n = int(grid_dict['first']['left_n'])
    left_step = float(grid_dict['first']['left_step'])
    right_n = int(grid_dict['first']['right_n'])
    right_step = float(grid_dict['first']['right_step'])

    if input_data_path is not None:
        kneadings_data, inits, nones, prev_config = get_kneadings_data(input_data_path)
        check_config_correspondence(prev_config, config, ('sf_grid',))
        _, _, params_x, params_y, _ = kneadings_data
    else:
        start_sys = so.FourBiharmonicPhaseOscillators(w, a, b, r)
        reduced_rhs_wrapper = start_sys.getReducedSystem
        reduced_jac_wrapper = start_sys.getReducedSystemJac
        get_params = start_sys.getParams
        set_params = start_sys.setParams

        if start_eq is not None:
            start = time.time()
            eq_grid = continue_equilibrium(reduced_rhs_wrapper, reduced_jac_wrapper, get_params, set_params,
                                           param_to_index, 'a', 'b',
                                           start_eq, up_n, down_n, left_n, right_n,
                                           up_step, down_step, left_step, right_step)
            sf_grid = get_eq_type_grid(eq_grid, up_n, down_n, left_n, right_n, sf.has1DUnstable, sf.STD_PRECISION)
            inits, nones = find_inits_for_equilibrium_grid(sf_grid, 3, up_n, down_n, left_n, right_n, sf.STD_PRECISION)
            params_x, params_y = generate_parameters((a, b), up_n, down_n, left_n, right_n,
                                                     up_step, down_step, left_step, right_step)
            end = time.time()
            print(f"Took {end - start}s ({datetime.timedelta(seconds=end - start)})")
        else:
            raise Exception("Start saddle-focus was not found!")

    return {'inits': inits, 'nones': nones, 'params_x': params_x, 'params_y': params_y, 'targetDir': 'output'}


@register(registry, 'worker', 'kneadings_fbpo')
def worker_kneadings_fbpo(config, initResult, timeStamp):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']
    def_params = [w, a, b, r]

    param_to_index = def_sys_dict['param_to_index']

    grid_dict = config['grid']
    left_n = grid_dict['first']['left_n']
    right_n = grid_dict['first']['right_n']
    up_n = grid_dict['second']['up_n']
    down_n = grid_dict['second']['down_n']
    param_x_name = grid_dict['first']['name']
    param_y_name = grid_dict['second']['name']

    kneadings_dict = config['kneadings_fbpo']
    dt = kneadings_dict['dt']
    n = kneadings_dict['n']
    stride = kneadings_dict['stride']
    kneadings_start = kneadings_dict['kneadings_start']
    kneadings_end = kneadings_dict['kneadings_end']

    inits = initResult['inits']
    nones = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']

    kneadings_weighted_sum_set = sweep(
        inits,
        nones,
        params_x,
        params_y,
        def_params,
        param_to_index,
        param_x_name,
        param_y_name,
        up_n,
        down_n,
        left_n,
        right_n,
        dt,
        n,
        stride,
        kneadings_start,
        kneadings_end
    )

    # print("Results:")
    kneadings_len = kneadings_end - kneadings_start + 1
    kneadings_records = ""
    for idx in range((left_n + right_n + 1) * (up_n + down_n + 1)):
        kneading_weighted_sum = kneadings_weighted_sum_set[idx]
        kneading_symbolic = convert_heavy_tail_to_sequence(kneading_weighted_sum, 4, kneadings_len)

        # print(f"a: {params_x[idx]:.15f}, "
        #       f"b: {params_y[idx]:.15f} => "
        #       f"{kneading_symbolic} (Raw: {kneading_weighted_sum})")
        kneadings_records = (kneadings_records + f"a: {params_x[idx]:.15f}, "
                                                 f"b: {params_y[idx]:.15f} => "
                                                 f"{kneading_symbolic} (Raw: {kneading_weighted_sum})\n")

    return {'kneadings_weighted_sum_set': kneadings_weighted_sum_set, 'kneadings_records': kneadings_records}


@register(registry, 'post', 'kneadings_fbpo')
def post_kneadings_fbpo(config, initResult, workerResult, grid, startTime):
    plot_params_dict = config['misc']['plot_params']
    font_size = plot_params_dict['font_size']

    grid_dict = config['grid']
    param_x_caption = grid_dict['first']['caption']
    left_n = grid_dict['first']['left_n']
    right_n = grid_dict['first']['right_n']
    param_y_caption = grid_dict['second']['caption']
    up_n = grid_dict['second']['up_n']
    down_n = grid_dict['second']['down_n']

    kneadings_dict = config['kneadings_fbpo']
    kneadings_start = kneadings_dict['kneadings_start']
    kneadings_end = kneadings_dict['kneadings_end']
    kneadings_len = kneadings_end - kneadings_start + 1

    inits = initResult['inits']
    nones = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']

    kneadings_weighted_sum_set = workerResult['kneadings_weighted_sum_set']
    kneadings_records = workerResult['kneadings_records']

    idxs_x = []
    idxs_y = []
    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            idxs_x.append(i)
            idxs_y.append(j)

    kneadings_data = [idxs_x,
                      idxs_y,
                      params_x,
                      params_y,
                      kneadings_weighted_sum_set]

    def set_color_map():
        return set_random_color_map(4, kneadings_len)
    fig = plot_mode_map(kneadings_data, set_color_map, param_x_caption, param_y_caption, font_size)
    plt.title(f"({param_x_caption}, {param_y_caption})-parameter sweep "
              f"of [{kneadings_start + 1}-{kneadings_end + 1}] length", fontsize=font_size)

    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        mode_map_data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    mode_map_data = mode_map_data.reshape((int(h), int(w), -1))

    # СОХРАНЕНИЕ

    h5py_outname = makeFinalOutname(config, initResult, "hdf5", startTime)
    with h5py.File(h5py_outname, 'w') as main_folder:
        kneadings_info = main_folder.create_group('kneadings_info')
        kneadings_info.create_dataset('kneadings_data', data=kneadings_data)
        kneadings_info.create_dataset('mode_map', data=mode_map_data)

        sf_grid_info = main_folder.create_group('sf_grid_info')
        sf_grid_info.create_dataset('inits', data=inits)
        sf_grid_info.create_dataset('nones', data=nones)
        # в эту же группу потом сохранять массив седлофокусов внутри подтетраэдра

        config_string = yaml.dump(config)
        main_folder.attrs['config'] = config_string
    print("Dataset successfully saved")

    txt_outname = makeFinalOutname(config, initResult, "txt", startTime)
    with open(txt_outname, 'w') as txt_output:
        txt_output.write(kneadings_records)
    print("Text records successfully saved")

    img_extension = config['output']['imageExtension']
    plot_outname = makeFinalOutname(config, initResult, img_extension, startTime)
    plt.savefig(plot_outname, dpi=600, bbox_inches='tight')
    plt.close()
    print("Mode map successfully saved")

    # пример восстановления картинки из hdf файла
    # plot_outname_jpg = makeFinalOutname(config, initResult, 'jpg', startTime)
    # with h5py.File(h5py_outname, 'r') as h5py_input:
    #     plt.imshow(h5py_input['mode_map'], interpolation='nearest')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(plot_outname_jpg, dpi=600, bbox_inches='tight')
