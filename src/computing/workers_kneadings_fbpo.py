import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from lib.computation_template.workers_sl import registry
from lib.computation_template.workers_utils import register, makeFinalOutname

import lib.eq_finder.SystOsscills as so
from src.system_analysis.get_inits import continue_equilibrium, get_saddle_foci_grid, find_inits_for_equilibrium_grid, generate_parameters
from src.mapping.convert import decimal_to_quaternary
from src.cuda_sweep.sweep_fbpo import sweep


@register(registry, 'init', 'kneadings_fbpo')
def init_kneadings_fbpo(config, timeStamp):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']

    param_to_index = config['misc']['param_to_index']
    start_eq = config['misc']['start_eq']

    grid_dict = config['grid']
    up_n = grid_dict['second']['up_n']
    up_step = grid_dict['second']['up_step']
    down_n = grid_dict['second']['down_n']
    down_step = grid_dict['second']['down_step']
    left_n = grid_dict['first']['left_n']
    left_step = grid_dict['first']['left_step']
    right_n = grid_dict['first']['right_n']
    right_step = grid_dict['first']['right_step']

    start_sys = so.FourBiharmonicPhaseOscillators(w, a, b, r)
    reduced_rhs_wrapper = start_sys.getReducedSystem
    reduced_jac_wrapper = start_sys.getReducedSystemJac
    get_params = start_sys.getParams
    set_params = start_sys.setParams

    if start_eq is not None:
        eq_grid = continue_equilibrium(reduced_rhs_wrapper, reduced_jac_wrapper, get_params, set_params,
                                       param_to_index, 'a', 'b',
                                       start_eq, up_n, down_n, left_n, right_n,
                                       up_step, down_step, left_step, right_step)
        sf_grid = get_saddle_foci_grid(eq_grid, up_n, down_n, left_n, right_n)
        inits, nones = find_inits_for_equilibrium_grid(sf_grid, 3, up_n, down_n, left_n, right_n, )
        params_x, params_y = generate_parameters((a, b), up_n, down_n, left_n, right_n,
                                            up_step, down_step, left_step, right_step)
    else:
        print("Start saddle-focus was not found")

    return {'inits': inits, 'nones': nones, 'params_x': params_x, 'params_y': params_y, 'targetDir': 'output'}


@register(registry, 'worker', 'kneadings_fbpo')
def worker_kneadings_fbpo(config, initResult, timeStamp):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']
    def_params = [w, a, b, r]

    param_to_index = config['misc']['param_to_index']

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

    print("Results:")
    for idx in range((left_n + right_n + 1) * (up_n + down_n + 1)):
        kneading_weighted_sum = kneadings_weighted_sum_set[idx]
        kneading_symbolic = decimal_to_quaternary(kneading_weighted_sum)

        print(f"a: {params_x[idx]:.6f}, "
              f"b: {params_y[idx]:.6f} => "
              f"{kneading_symbolic} (Raw: {kneading_weighted_sum})")

    return kneadings_weighted_sum_set


@register(registry, 'post', 'kneadings_fbpo')
def post_kneadings_fbpo(config, initResult, workerResult, grid, startTime):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']

    grid_dict = config['grid']
    up_n = grid_dict['second']['up_n']
    up_step = grid_dict['second']['up_step']
    down_n = grid_dict['second']['down_n']
    down_step = grid_dict['second']['down_step']
    left_n = grid_dict['first']['left_n']
    left_step = grid_dict['first']['left_step']
    right_n = grid_dict['first']['right_n']
    right_step = grid_dict['first']['right_step']

    kneadings_weighted_sum_set = workerResult

    x_count = left_n + right_n + 1
    y_count = up_n + down_n + 1
    x_start = a - left_n * left_step
    x_end = a + right_n * right_step
    y_start = b - down_n * down_step
    y_end = b + up_n * up_step

    colorMapLevels = 2 ** 8
    blue = np.linspace(0.01, 1.0, colorMapLevels)
    red = 1 - blue
    green = np.random.random(colorMapLevels) * 0.8
    # green = np.linspace(0.8, 1.0, colorMapLevels)
    RGB = np.column_stack((red, green, blue))
    custom_cmap = ListedColormap(RGB)

    # нормализация

    plt.figure(figsize=(8, 8))
    plt.imshow(
        np.reshape(kneadings_weighted_sum_set, (x_count, y_count), 'F'),
        extent=[x_start, x_end, y_start, y_end],
        cmap=custom_cmap,
        vmin=-0.1,
        vmax=1,
        origin='lower',
        aspect='auto'
    )
    plt.xlabel('Параметр a')
    plt.ylabel('Параметр b')
    plt.title('Карта режимов')

    outFileExtension = config['output']['imageExtension']
    outname = makeFinalOutname(config, initResult, outFileExtension, startTime)
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    print("Mode map successfully saved")