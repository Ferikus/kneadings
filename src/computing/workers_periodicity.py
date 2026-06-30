import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import io

from src.computing.engines import (get_kneadings_data, get_config_data, check_config_correspondence,
                                   save_data)
from src.cuda_sweep.sweep_period import sweep_period_complexity
from src.routing.route_exploring import get_grid_points_along_line

### to connect with workers file
from lib.computation_template.workers_utils import register, makeFinalOutname
from src.computing.workers import registry


@register(registry, 'worker', 'periodicity')
def worker_periodicity_fbpo(config, initResult, timeStamp):
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

    periodicity_dict = config['periodicity']
    dt = periodicity_dict['dt']
    n = periodicity_dict['n']
    stride = periodicity_dict['stride']
    kneadings_start = periodicity_dict['kneadings_start']
    kneadings_end = periodicity_dict['kneadings_end']
    input_data_path = periodicity_dict['input_data']

    inits = initResult['inits']
    nones = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']
    inner_sf_set = initResult['inner_sf_set']

    if input_data_path is not None:
        prev_config = get_config_data(input_data_path)
        check_config_correspondence(prev_config, config, ('sf_grid', 'periodicity',))
        periods_data = get_kneadings_data(input_data_path)
        _, _, _, _, period_set = periods_data
    else:
        def_params = [w, a, b, r]
        period_set = sweep_period_complexity(
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
            kneadings_end,
            inner_sf_set
        )

    return {'period_set': period_set}


@register(registry, 'post', 'periodicity')
def post_periodicity_fbpo(config, initResult, workerResult, grid, startTime):
    grid_dict = config['grid']
    param_x_caption = grid_dict['first']['caption']
    param_x_name = grid_dict['first']['name']
    left_n = grid_dict['first']['left_n']
    right_n = grid_dict['first']['right_n']
    param_y_caption = grid_dict['second']['caption']
    param_y_name = grid_dict['second']['name']
    up_n = grid_dict['second']['up_n']
    down_n = grid_dict['second']['down_n']

    inits = initResult['inits']
    nones = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']
    inner_sf_set = initResult['inner_sf_set']

    periodicity_dict = config['periodicity']
    kneadings_start = periodicity_dict['kneadings_start']
    kneadings_end = periodicity_dict['kneadings_end']

    plot_settings = config['misc']['plot_settings']['default']

    period_set = workerResult['period_set']

    idxs_x = []
    idxs_y = []
    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            idxs_x.append(i)
            idxs_y.append(j)

    periods_data = [idxs_x, idxs_y, params_x, params_y, period_set]

    plot_data = np.ma.masked_where(period_set < 0, period_set)
    custom_cmap = plt.get_cmap('gist_rainbow').copy()
    custom_cmap.set_under('black')
    custom_cmap.set_bad('gray')

    unique_params_x = np.unique(params_x)
    unique_params_y = np.unique(params_y)
    param_x_count = len(unique_params_x)
    param_y_count = len(unique_params_y)

    mpl.rcParams.update(plot_settings)
    fig = plt.figure()
    plt.pcolormesh(unique_params_x, unique_params_y,
                   plot_data.reshape((param_y_count, param_x_count)),
                   cmap=custom_cmap,
                   shading='nearest',
                   vmin=1.0, vmax=4.0,  #np.max(valid_periods) if valid_periods.size > 0 else 2.0
                   rasterized=True)
    plt.xlim(unique_params_x.min(), unique_params_x.max())
    plt.ylim(unique_params_y.min(), unique_params_y.max())
    plt.axis('scaled')
    plt.xlabel(f'${param_x_caption}$')
    plt.ylabel(f'${param_y_caption}$')
    plt.tick_params(axis='x')
    plt.tick_params(axis='y')
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.title(f"(${param_x_caption}$, ${param_y_caption}$)-parameter period complexity map\n"
              f"for [{kneadings_start + 1}-{kneadings_end + 1}] kneadings length")

    # 1. Создаем нормализацию для сопоставления значений 1..4 с цветами палитры
    legend_vals = np.unique([val for val in period_set if val > 0]).astype(int)
    norm = mpl.colors.Normalize(vmin=1.0, vmax=4.0)
    # 2. Формируем список плашек для каждого периода
    legend_patches = [
        mpatches.Patch(color='black', label='Irregular'),
        mpatches.Patch(color='gray', label='Error'),
        *[mpatches.Patch(color=custom_cmap(norm(p)), label=f'{p}') for p in legend_vals]
    ]
    plt.legend(
        handles=legend_patches,
        title="Period complexity",
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.
    )

    def onclick(event):
        xdata = event.xdata
        ydata = event.ydata

        pts_idxs, pts_coords, pts_vals = get_grid_points_along_line(periods_data, (xdata, ydata), (xdata, ydata), 1)
        pt_idx, pt_coords, period = pts_idxs[0], pts_coords[0], pts_vals[0]

        print(f"Clicked at node {pt_idx} with parameters {param_x_name}={pt_coords[0]:.15f}, {param_y_name}={pt_coords[1]:.15f}, "
              f"period {period}")

    fig.canvas.mpl_connect('button_press_event', onclick)
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
    mode_map_data = mode_map_data.reshape((h, w, -1))

    # SAVING

    kneadings_records = ""
    for idx in range((left_n + right_n + 1) * (up_n + down_n + 1)):
        period = period_set[idx]

        if period < 0:  # CannotGetPeriodError
            regime_label = "ERROR"
        elif period == 0:
            regime_label = "IRREGULAR"
        else:
            regime_label = f"REGULAR (period={period})"

        kneadings_records += (f"{param_x_name}: {params_x[idx]:.15f}, "
                              f"{param_y_name}: {params_y[idx]:.15f} => "
                              f"{regime_label}\n")

    txt_outname = makeFinalOutname(config, initResult, "txt", startTime)
    with open(txt_outname, 'w', encoding="utf-8") as txt_output:
        txt_output.write(kneadings_records)
    print("Text records successfully saved")

    img_extension = config['output']['imageExtension']
    plot_outname = makeFinalOutname(config, initResult, img_extension, startTime)
    fig.savefig(plot_outname, bbox_inches='tight')
    plt.close(fig)
    print("Periodicity map successfully saved")

    hdf5_outname = makeFinalOutname(config, initResult, "hdf5", startTime)
    save_data(hdf5_outname, periods_data, kneadings_records, mode_map_data, inits, nones,
              inner_sf_set, config)
    print("Dataset successfully saved")