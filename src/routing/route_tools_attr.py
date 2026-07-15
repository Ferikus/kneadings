from itertools import groupby

from src.plotting.plot_attractors import plot_attractors_plt, get_face_sf_trajectory, plot_thetrahedron
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


def plot_target_attractors_attr(config, views, saving_directory, plotting_data, convert_func):
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
    dt = kneadings_dict['dt']
    n = kneadings_dict['n']

    route_dict = config['route']
    # dt = route_dict['dt']
    # n = route_dict['n']
    # skip = route_dict['skip']
    skip = 0

    img_ext = config['output']['imageExtension']

    plot_settings = config['misc']['plot_settings']['default']
    plt.rcParams.update(plot_settings)

    params = [w, a, b, r]
    kneadings_len = kneadings_end - kneadings_start + 1

    print("Generating phase portraits...")
    for i, rep_pt in enumerate(plotting_data):
        param_x, param_y = rep_pt['coords']
        val = rep_pt['val']

        val_converted = convert_func(val, 4, kneadings_len)

        params[param_to_index[param_x_name]] = param_x
        params[param_to_index[param_y_name]] = param_y

        traj_t0 = np.array(get_face_sf_trajectory(params, n, dt))
        traj_t1 = np.array(list(map(t, traj_t0.T))).T
        traj_t2 = np.array(list(map(t, traj_t1.T))).T
        trajs = [traj_t0, traj_t2, traj_t1]

        symm_type, symm_dist_t1, symm_dist_t2 = getSymmetryTypeByHistogram(traj_t0.T, nLevels=256, precision=0.1)

        print(f"Generating phase portrait for point {i}: ({param_x:.13f}, {param_y:.13f}) at sequence {val_converted}\n"
              f"with symmetry T{symm_type} (dist_t1: {symm_dist_t1}, dist_t2: {symm_dist_t2})")
        plot_attractors_plt(
            trajs,
            views=views,
            plot_placeholder=plot_thetrahedron,
            start_pt=skip,
            directory=saving_directory,
            point_name=f"attr_{i}_{val_converted}_T{symm_type}",
            img_ext=img_ext
        )
