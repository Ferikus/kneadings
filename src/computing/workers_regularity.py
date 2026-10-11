import numpy as np
import matplotlib.pyplot as plt
import io

from src.computing.engines import save_data
from src.cuda_sweep.sweep_period import sweep_period_metrics, unpack_period_metrics
from src.routing.route_exploring import get_grid_points_along_line
from src.plotting.plot_mode_map import plot_regularity_data
from src.computing.workers import registry, make_kneadings_based_worker

### to connect with workers file
from lib.computation_template.workers_utils import register, makeFinalOutname

worker_regularity = make_kneadings_based_worker('regularity', sweep_period_metrics)


@register(registry, 'post', 'regularity')
def post_regularity(config, initResult, workerResult, grid, startTime):
    grid_dict = config['grid']
    param_x_caption = grid_dict['first']['caption']
    param_x_name = grid_dict['first']['name']
    left_n = grid_dict['first']['left_n']
    right_n = grid_dict['first']['right_n']
    param_y_caption = grid_dict['second']['caption']
    param_y_name = grid_dict['second']['name']
    up_n = grid_dict['second']['up_n']
    down_n = grid_dict['second']['down_n']

    regularity_dict = config['regularity']
    kneadings_start = regularity_dict['kneadings_start']
    kneadings_end = regularity_dict['kneadings_end']

    plot_settings = config['misc']['plot_settings']['default']

    inits = initResult['inits']
    nones = initResult['nones']
    params_x = initResult['params_x']
    params_y = initResult['params_y']
    inner_sf_set = initResult['inner_sf_set']

    packed_set = workerResult['task_core_data']
    regularity_set, complexity_set, periodicity_set = unpack_period_metrics(packed_set)

    idxs_x = []
    idxs_y = []
    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            idxs_x.append(i)
            idxs_y.append(j)

    regularity_data = [idxs_x, idxs_y, params_x, params_y, regularity_set]

    fig = plot_regularity_data(
        regularity_data, param_x_caption, param_y_caption, plot_settings
    )
    plt.title(f"(${param_x_caption}$, ${param_y_caption}$)-parameter regularity map\n"
              f"for [{kneadings_start + 1}-{kneadings_end + 1}] kneadings length")

    def onclick(event):
        xdata = event.xdata
        ydata = event.ydata

        pts_idxs, pts_coords, pts_vals = get_grid_points_along_line(regularity_data, (xdata, ydata), (xdata, ydata), 1)
        pt_idx, pt_coords, regularity = pts_idxs[0], pts_coords[0], pts_vals[0]

        regularity_desc = bool(regularity)
        print(f"Clicked at node {pt_idx} with parameters {param_x_name}={pt_coords[0]:.15f}, {param_y_name}={pt_coords[1]:.15f}, "
              f"regularity {regularity_desc}")

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
        reg = regularity_set[idx]

        if reg < 0:
            regime_label = f"ERROR (code={reg})"
        elif reg == 0:
            regime_label = "IRREGULAR"
        else:
            regime_label = "REGULAR"

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
    print("Regularity map successfully saved")

    hdf5_data = [idxs_x, idxs_y, params_x, params_y, packed_set]
    hdf5_outname = makeFinalOutname(config, initResult, "hdf5", startTime)
    save_data(hdf5_outname, hdf5_data, txt_records, mode_map_data, inits, nones,
              inner_sf_set, config)
    print("Dataset successfully saved")