import h5py
import numpy as np
import matplotlib.pyplot as plt
import io

from src.computing.engines import (get_config_data, check_config_correspondence, save_data)
from src.symmetry.detectives import sweep_detective, makeRuleOne
from src.plotting.plot_mode_map import plot_detectives_data

### to connect with workers file
from lib.computation_template.workers_utils import register, makeFinalOutname
from src.computing.workers import registry


@register(registry, 'worker', 'symmetry_detectives')
def worker_symmetry_detectives(config, initResult, timeStamp):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']
    param_to_index = def_sys_dict['param_to_index']

    grid_dict = config['grid']
    left_n = grid_dict['first']['left_n']
    right_n = grid_dict['first']['right_n']
    up_n = grid_dict['second']['up_n']
    down_n = grid_dict['second']['down_n']
    param_x_name = grid_dict['first']['name']
    param_y_name = grid_dict['second']['name']

    detectives_params = config['symmetry_detectives']
    dt = float(detectives_params['dt'])
    nStepsSkip = int(detectives_params['nStepsSkip'])
    nStepsAttractor = int(detectives_params['nStepsAttractor'])
    input_data_path = detectives_params['input_data']

    inits = initResult['inits']
    nones = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']

    if input_data_path is not None:
        prev_config = get_config_data(input_data_path)
        check_config_correspondence(prev_config, config, ('sf_grid', 'symmetry_detectives',))

        with h5py.File(input_data_path, 'r') as input_file:
            data = input_file['symmetry_detectives_info']['symmetry_detectives_data'][()]
        detectives_outputs = np.column_stack((data[4], data[5], data[6], data[7]))
    else:
        def_params = [w, a, b, r]
        detectives_outputs = sweep_detective(
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
            nStepsSkip,
            nStepsAttractor
        )

    return {'detectives_outputs': detectives_outputs}


@register(registry, 'post', 'symmetry_detectives')
def post_symmetry_detectives(config, initResult, workerResult, grid, startTime):
    grid_dict = config['grid']
    param_x_caption = grid_dict['first']['caption']
    left_n = grid_dict['first']['left_n']
    right_n = grid_dict['first']['right_n']
    param_y_caption = grid_dict['second']['caption']
    up_n = grid_dict['second']['up_n']
    down_n = grid_dict['second']['down_n']

    plot_settings = config['misc']['plot_settings']['default']

    inits = initResult['inits']
    nones = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']
    inner_sf_set = initResult['inner_sf_set']

    detectives_outputs = workerResult['detectives_outputs']
    detectives_outputs = np.reshape(detectives_outputs, (-1, 4))

    idxs_x = []
    idxs_y = []
    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            idxs_x.append(i)
            idxs_y.append(j)

    detectives_data = [idxs_x,
                      idxs_y,
                      params_x,
                      params_y,
                      detectives_outputs]

    fig = plot_detectives_data(detectives_data, makeRuleOne(-1.5, -0.5), param_x_caption, param_y_caption, plot_settings)
    plt.title(f"(${param_x_caption}$, ${param_y_caption}$)-parameter symmetry map")
    plt.tight_layout()
    plt.show()

    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        mode_map_data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    save_dpi = plot_settings['savefig.dpi']
    w_inch, h_inch = fig.get_size_inches()
    w = int(w_inch * save_dpi)
    h = int(h_inch * save_dpi)
    mode_map_data = mode_map_data.reshape((int(h), int(w), -1))

    # SAVING

    # jsonData = {'idxs_x': [1],
    #             'idxs_y': [1],
    #             'params_x': list(params_x),
    #             'params_y': list(params_y),
    #             'detectives_outputs': list(detectives_outputs)}

    # with open(f'for-plot-{timeStamp}.json', 'w') as f:
    #     json.dump(jsonData, f)

    # fileContent = ['# idx param_x_name params_x[idx] param_y_name params_y[idx] errCode log10_D01 log10_D02 log10_D03\n']
    fileContent = ['# idx params_x[idx] params_y[idx] errCode log10_D01 log10_D02 log10_D03\n']

    for idx in range((left_n + right_n + 1) * (up_n + down_n + 1)):
        curEntry = detectives_outputs[idx]
        # fileContent.append(f"{idx} {param_x_name} {params_x[idx]} {param_y_name} {params_y[idx]} {curEntry[0]} {np.log10(curEntry[1])} {np.log10(curEntry[2])}  {np.log10(curEntry[3])}\n")
        fileContent.append(f"{idx} {params_x[idx]} {params_y[idx]} {curEntry[0]} {np.log10(curEntry[1])} {np.log10(curEntry[2])}  {np.log10(curEntry[3])}\n")

    txt_outname = makeFinalOutname(config, initResult, "txt", startTime)
    with open(txt_outname, 'w') as txt_output:
        txt_output.writelines(fileContent)
    print("Text records successfully saved")

    img_extension = config['output']['imageExtension']
    plot_outname = makeFinalOutname(config, initResult, img_extension, startTime)
    fig.savefig(plot_outname, bbox_inches='tight')
    plt.close(fig)
    print("Mode map successfully saved")

    # detectives_array_gpu = cuda.device_array(total_parameter_space_size*DETECTIVE_ENTRY_SIZE)
    # DETECTIVE_ENTRY_SIZE = 4
    # разделить на DETECTIVE_ENTRY_SIZE массивов при сохранении и потом склеивать в один, когда открываем

    out0 = detectives_outputs[:, 0]
    out1 = detectives_outputs[:, 1]
    out2 = detectives_outputs[:, 2]
    out3 = detectives_outputs[:, 3]

    detectives_data_save = [
        np.array(idxs_x),
        np.array(idxs_y),
        np.array(params_x),
        np.array(params_y),
        out0,
        out1,
        out2,
        out3
    ]

    hdf5_outname = makeFinalOutname(config, initResult, "hdf5", startTime)
    save_data(hdf5_outname, detectives_data_save, fileContent, mode_map_data, inits, nones, inner_sf_set, config)
    print("Dataset successfully saved")
