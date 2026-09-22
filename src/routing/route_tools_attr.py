import os
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby

from src.plotting.plot_attractors import (plot_attractors_plt, get_face_sf_trajectory_3d_4d,
                                          plot_thetrahedron, save_trajectory_to_txt, save_trajectory_to_npz)
from src.routing.route_exploring import *
from src.symmetry.histogram import getSymmetryTypeByHistogram
from src.system_analysis.taskutils import t


def get_target_points_attr(idxs, coords, vals):
    rep_pts = []

    for val, group in groupby(enumerate(vals), key=lambda x: x[1]):
        if val < 0:
            continue

        group_indices = [i for i, _ in group]
        mid_idx = group_indices[len(group_indices) // 2]

        rep_pts.append(make_target_point(idxs, coords, mid_idx, val))

    slice_coords = [pt['coords'] for pt in rep_pts]

    return rep_pts, slice_coords


def _process_single_point(task_args):
    (i, rep_pt, base_params, param_to_index, param_x_name, param_y_name,
     kneadings_len, convert_func, n, dt, skip, views, save_dir, img_ext) = task_args

    param_x, param_y = rep_pt['coords']
    val = rep_pt['val']
    val_converted = convert_func(val, 4, kneadings_len)

    params = list(base_params)
    params[param_to_index[param_x_name]] = param_x
    params[param_to_index[param_y_name]] = param_y

    # compute 4d and then 3d based on reduction formulas
    traj_t0, traj_4d = get_face_sf_trajectory_3d_4d(params, n, dt)

    traj_t1 = np.array(list(map(t, traj_t0.T))).T
    traj_t2 = np.array(list(map(t, traj_t1.T))).T
    traj_t3 = np.array(list(map(t, traj_t2.T))).T
    trajs = [traj_t0, traj_t2, traj_t1, traj_t3]  # red, green, blue, pink

    symm_type, symm_dist_t1, symm_dist_t2 = getSymmetryTypeByHistogram(traj_t0.T, nLevels=384, precision=0.1)

    point_name = f"attr_{i}_{val_converted}_T{symm_type}"

    data_dir = os.path.join(save_dir, "data")

    save_trajectory_to_npz(
        os.path.join(data_dir, f"{point_name}_3d.npz"),
        params,
        traj_t0[:, skip:],
        ['x', 'y', 'z']
    )
    save_trajectory_to_npz(
        os.path.join(data_dir, f"{point_name}_4d.npz"),
        params,
        traj_4d[:, skip:],
        ['phi0', 'phi1', 'phi2', 'phi3']
    )

    print(f"[{i}] ({param_x:.6f}, {param_y:.6f}) seq={val_converted} T{symm_type} (dist_t1: {symm_dist_t1:.4f}, dist_t2: {symm_dist_t2:.4f})")

    fig, _ = plot_attractors_plt(
        trajs,
        views=views,
        plot_placeholder=plot_thetrahedron,
        start_pt=skip,
        directory=save_dir,
        point_name=point_name,
        img_ext=img_ext
    )
    plt.close(fig)


def plot_target_attractors_attr(config, views, save_dir, plotting_data, convert_func):
    def_sys_dict = config['defaultSystem']
    w = def_sys_dict['w']
    a = def_sys_dict['a']
    b = def_sys_dict['b']
    r = def_sys_dict['r']
    param_to_index = def_sys_dict['param_to_index']

    grid_dict = config['grid']
    param_x_name = grid_dict['first']['name']
    param_y_name = grid_dict['second']['name']

    kneadings_dict = config['kneadings']
    kneadings_start = kneadings_dict['kneadings_start']
    kneadings_end = kneadings_dict['kneadings_end']
    # dt = kneadings_dict['dt']
    # n = kneadings_dict['n']

    route_dict = config['route']
    skip = route_dict['skip']
    dt = route_dict['dt']
    n = route_dict['n'] + skip
    # skip = 0

    img_ext = config['output']['imageExtension']

    plot_settings = config['misc']['plot_settings']['default']
    plt.rcParams.update(plot_settings)

    params = [w, a, b, r]
    kneadings_len = kneadings_end - kneadings_start + 1

    print("Generating phase portraits...")
    # for i, rep_pt in enumerate(plotting_data):
    #     param_x, param_y = rep_pt['coords']
    #     val = rep_pt['val']
    #
    #     val_converted = convert_func(val, 4, kneadings_len)
    #
    #     params[param_to_index[param_x_name]] = param_x
    #     params[param_to_index[param_y_name]] = param_y
    #
    #     traj_t0 = np.array(get_face_sf_trajectory(params, n, dt))
    #     traj_t1 = np.array(list(map(t, traj_t0.T))).T
    #     traj_t2 = np.array(list(map(t, traj_t1.T))).T
    #     trajs = [traj_t0, traj_t2, traj_t1]
    #
    #     symm_type, symm_dist_t1, symm_dist_t2 = getSymmetryTypeByHistogram(traj_t0.T, nLevels=256, precision=0.1)
    #
    #     print(f"Generating phase portrait for point {i}: ({param_x:.13f}, {param_y:.13f}) at sequence {val_converted}\n"
    #           f"with symmetry T{symm_type} (dist_t1: {symm_dist_t1}, dist_t2: {symm_dist_t2})")
    #     plot_attractors_plt(
    #         trajs,
    #         views=views,
    #         plot_placeholder=plot_thetrahedron,
    #         start_pt=skip,
    #         directory=saving_directory,
    #         point_name=f"attr_{i}_{val_converted}_T{symm_type}",
    #         img_ext=img_ext
    #     )
    tasks = [
        (i, rep_pt, params, param_to_index, param_x_name, param_y_name,
         kneadings_len, convert_func, n, dt, skip, views, save_dir, img_ext)
        for i, rep_pt in enumerate(plotting_data)
    ]

    max_workers = os.cpu_count()
    print(f"Generating phase portraits in parallel using {max_workers} processes...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_process_single_point, tasks))