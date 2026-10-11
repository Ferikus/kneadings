import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

from src.computing.engines import save_data
from src.cuda_sweep.sweep_period import sweep_period_metrics, unpack_period_metrics
from src.routing.route_exploring import get_grid_points_along_line
from src.plotting.plot_mode_map import plot_complexity_data

### to connect with workers file
from lib.computation_template.workers_utils import register, makeFinalOutname
from src.computing.workers import registry, make_kneadings_based_worker

worker_complexity = make_kneadings_based_worker('complexity', sweep_period_metrics)


@register(registry, 'post', 'complexity')
def post_complexity(config, initResult, workerResult, grid, startTime):
    grid_dict = config['grid']
    param_x_caption = grid_dict['first']['caption']
    param_x_name = grid_dict['first']['name']
    left_n = grid_dict['first']['left_n']
    right_n = grid_dict['first']['right_n']
    param_y_caption = grid_dict['second']['caption']
    param_y_name = grid_dict['second']['name']
    up_n = grid_dict['second']['up_n']
    down_n = grid_dict['second']['down_n']

    task_dict = config['complexity']
    kneadings_start = task_dict['kneadings_start']
    kneadings_end = task_dict['kneadings_end']

    plot_settings = config['misc']['plot_settings']['default']

    inits = initResult['inits']
    nones = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']
    inner_sf_set = initResult['inner_sf_set']

    packed_set = workerResult['task_core_data']
    _, complexity_set, _ = unpack_period_metrics(packed_set)

    idxs_x = []
    idxs_y = []
    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            idxs_x.append(i)
            idxs_y.append(j)

    complexity_data = [idxs_x, idxs_y, params_x, params_y, complexity_set]

    fig = plot_complexity_data(
        complexity_data, param_x_caption, param_y_caption, plot_settings
    )
    plt.title(f"(${param_x_caption}$, ${param_y_caption}$)-parameter kneadings complexity map\n"
              f"for [{kneadings_start + 1}-{kneadings_end + 1}] kneadings length")

    def onclick(event):
        xdata = event.xdata
        ydata = event.ydata

        pts_idxs, pts_coords, pts_vals = get_grid_points_along_line(complexity_data, (xdata, ydata), (xdata, ydata), 1)
        pt_idx, pt_coords, complexity = pts_idxs[0], pts_coords[0], pts_vals[0]

        print(f"Clicked at node {pt_idx} with parameters {param_x_name}={pt_coords[0]:.15f}, {param_y_name}={pt_coords[1]:.15f}, "
              f"complexity {complexity}")

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

    # saving
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        mode_map_data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    save_dpi = plot_settings['savefig.dpi']
    w_inch, h_inch = fig.get_size_inches()
    w = int(w_inch * save_dpi)
    h = int(h_inch * save_dpi)
    mode_map_data = mode_map_data.reshape((h, w, -1))

    txt_records = ""
    for idx in range((left_n + right_n + 1) * (up_n + down_n + 1)):
        complexity = complexity_set[idx]

        if complexity < 0:  # CannotGetPeriodError
            regime_label = f"ERROR (code={complexity})"
        elif complexity > 0:
            regime_label = f"REGULAR (period={complexity})"
        else:
            regime_label = f"Complexity equals zero! Why?"

        txt_records += (f"{param_x_name}: {params_x[idx]:.15f}, "
                              f"{param_y_name}: {params_y[idx]:.15f} => "
                              f"{regime_label}\n")

    txt_outname = makeFinalOutname(config, initResult, "txt", startTime)
    with open(txt_outname, 'w', encoding="utf-8") as txt_output:
        txt_output.write(txt_records)
    print("Text records successfully saved")

    img_extension = config['output']['imageExtension']
    plot_outname = makeFinalOutname(config, initResult, img_extension, startTime)
    fig.savefig(plot_outname, bbox_inches='tight')
    plt.close(fig)
    print("Complexity map successfully saved")

    hdf5_data = [idxs_x, idxs_y, params_x, params_y, packed_set]
    hdf5_outname = makeFinalOutname(config, initResult, "hdf5", startTime)
    save_data(hdf5_outname, hdf5_data, txt_records, mode_map_data, inits, nones,
              inner_sf_set, config)
    print("Dataset successfully saved")