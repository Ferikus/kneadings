import plotly.graph_objects as go
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
from itertools import groupby

import lib.eq_finder.systems_fun as sf
import lib.eq_finder.SystOsscills as so
from tests.cpu_version import bary_expansion, get_domain_num, stepper_rk4
from src.system_analysis.thetrahedron import *
from src.system_analysis.get_inits import find_equilibrium_by_guess


def color_pt_by_domain(pt):
    colorScheme = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange'}
    return colorScheme[get_domain_num(bary_expansion(pt))]


def find_init_pts(sys):
    """Возвращает начальные условия для седло-фокуса"""
    bounds = [(-0.1, 2 * np.pi + 0.1)] * 2
    borders = [(-1e-15, 2 * np.pi + 1e-15)] * 2

    # первые две функции -- общая система, вторые две -- в которой ищем с.р., дальше функция приведения
    equilibria = sf.findEquilibria(lambda psis: sys.getReducedSystem(psis),
                                   lambda psis: sys.getReducedSystemJac(psis),
                                   lambda psis: sys.getRestriction(psis),
                                   lambda psis: sys.getRestrictionJac(psis),
                                   lambda phi: np.concatenate([[0.], phi]), bounds, borders,
                                   sf.ShgoEqFinder(1000, 1, 1e-10),
                                   sf.STD_PRECISION)

    for eq in equilibria:  # перебираем все с.р., которые были найдены
        if sf.has1DUnstable(eq, sf.STD_PRECISION):
            start_eq = np.array(eq.coordinates)
            if sf.is3DSaddleWith1dU(eq, sf.STD_PRECISION):
                print(f"Found saddle {start_eq}")
            elif sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION):
                print(f"Found saddle-focus {start_eq}")
            if sf.getInitPointsOnUnstable1DSeparatrix(eq, sf.pickCirSeparatrix, sf.STD_PRECISION):
                init_pt = sf.getInitPointsOnUnstable1DSeparatrix(eq, sf.pickCirSeparatrix, sf.STD_PRECISION)[0]
                print(f"with starting point {init_pt}")
                print(f"Eigenvalues: {eq.eigenvalues}")
                return init_pt


def compute_trajectory(y_curr, params, n, dt):
    """Считает траекторию при заданных начальных условиях"""
    trajectory = np.zeros((n, 3))
    trajectory[0] = y_curr
    last_n = 0

    for i in range(1, n):
        y_curr = stepper_rk4(params, y_curr, dt)
        trajectory[i] = y_curr

        for k in range(3):
            if y_curr[k] > 10 or y_curr[k] < -10:
                print('InfinityError')
                return trajectory

        last_n = i

    trajectory = (trajectory[:last_n]).T
    return trajectory


def get_eqs_on_inv_plane(params):
    sys = so.FourBiharmonicPhaseOscillators(*params)
    bounds = [(-0.1, 2 * np.pi + 0.1)] * 2
    borders = [(-1e-15, 2 * np.pi + 1e-15)] * 2

    ps = sf.STD_PRECISION

    # первые две функции -- общая система, вторые две -- в которой ищем с.р., дальше функция приведения
    eq_list = sf.findEquilibria(lambda psis: sys.getReducedSystem(psis),
                                lambda psis: sys.getReducedSystemJac(psis),
                                lambda psis: sys.getRestriction(psis),
                                lambda psis: sys.getRestrictionJac(psis),
                                lambda phi: np.concatenate([[0.], phi]), bounds, borders,
                                sf.ShgoEqFinder(1000, 1, 1e-10),
                                ps)

    eq_coords_list = []
    for eq in eq_list:
        if sf.isInCIR(eq.coordinates):
            eq_coords_list.append(eq.coordinates)

    all_symm_eqs = [point for eq in eq_coords_list
                    for point in sf.generateSymmetricPoints(eq)]

    return all_symm_eqs


def get_trajectory(params, n, dt):
    sys = so.FourBiharmonicPhaseOscillators(*params)
    init_psis = list(find_init_pts(sys))
    traj = compute_trajectory(init_psis, params, n, dt)
    return traj


def get_trajectories(params_set, n, dt):
    trajs = []
    for params in params_set:
        trajs.append(get_trajectory(params, n, dt))
    return trajs


def plot_attractors_go(trajs, eq_coords_list, rhs_jac, start_pt=0, ps=sf.STD_PRECISION):
    """Функция построения двух аттракторов для их сравнения:
    на вход требуется две траектории и точка, с которой начнётся их построение"""

    opacity = 0.5
    colors = cycle(['orange', 'purple', 'green'])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=tetraX, y=tetraY, z=tetraZ, marker=dict(size=3), line=dict(color="black"), opacity=opacity))
    fig.add_trace(go.Scatter3d(x=frameX, y=frameY, z=frameZ, marker=dict(size=3), line=dict(color="black"), opacity=opacity))
    fig.add_trace(go.Scatter3d(x=plankX, y=plankY, z=plankZ, marker=dict(size=3), line=dict(color="black"), opacity=opacity))

    for traj in trajs:
        if start_pt == 0:
            fig.add_trace(go.Scatter3d(x=[traj[0][0]], y=[traj[1][0]], z=[traj[2][0]], marker=dict(size=5, color='green'), mode='markers'))
        fig.add_trace(go.Scatter3d(x=traj[0][start_pt:], y=traj[1][start_pt:], z=traj[2][start_pt:], marker=dict(size=1), line=dict(color=next(colors))))

    for eq_coords in eq_coords_list:
        eq_obj = sf.getEquilibriumInfo(eq_coords, rhs_jac)
        eq_color = 'gray'
        if sf.is3DSaddleWith1dU(eq_obj, ps) or sf.is3DSaddleWith1dS(eq_obj, ps):
            eq_color = 'red'
        elif sf.is3DSaddleFocusWith1dU(eq_obj, ps) or sf.is3DSaddleFocusWith1dS(eq_obj, ps):
            eq_color = 'magenta'
        elif sf.isSink(eq_obj, ps):
            eq_color = 'blue'

        fig.add_trace(go.Scatter3d(x=[eq_obj.coordinates[0]], y=[eq_obj.coordinates[1]], z=[eq_obj.coordinates[2]],
                                   marker=dict(size=4), line=dict(color=eq_color), opacity=1))

    camera = dict(
        up=dict(x=0.0, y=0.0, z=1.5),
        center=dict(x=0.0, y=0.0, z=0.2),
        eye=dict(x=0.5, y=0.0, z=0.2)
    )
    fig.update_layout(showlegend=False,
                      scene=dict(xaxis=dict(backgroundcolor="white"),
                                 yaxis=dict(backgroundcolor="white"),
                                 zaxis=dict(backgroundcolor="white")))
    fig.update_layout(scene_camera=camera)

    fig.show()


# def plot_all_equilibrium(ax, eq_coords_list, rhs_jac, ps=sf.STD_PRECISION):
#     for eq_coords in eq_coords_list:
#         eq_obj = sf.getEquilibriumInfo(eq_coords, rhs_jac)
#
#         eq_color = 'gray'
#         eq_marker = 'o'
#
#         if sf.is3DSaddleWith1dU(eq_obj, ps) or sf.is3DSaddleWith1dS(eq_obj, ps):
#             eq_color = 'red'
#             if sf.is3DSaddleWith1dU(eq_obj, ps):
#                 eq_marker = '^'
#             else:
#                 eq_marker = 'v'
#         elif sf.is3DSaddleFocusWith1dU(eq_obj, ps) or sf.is3DSaddleFocusWith1dS(eq_obj, ps):
#             eq_color = 'magenta'
#             if sf.is3DSaddleFocusWith1dU(eq_obj, ps):
#                 eq_marker = '^'
#             else:
#                 eq_marker = 'v'
#         elif sf.isSink(eq_obj, ps):
#             eq_color = 'blue'
#
#         alpha = 1.0
#         if sf.isInCIR(eq_coords, strictly=True):
#             alpha = 0.5
#
#         ax.scatter(eq_obj.coordinates[0], eq_obj.coordinates[1], eq_obj.coordinates[2],
#                    s=50, c=eq_color, marker=eq_marker, alpha=alpha)


def plot_saddle_at_sepbif(ax, trajs, params1, params2, threshold, n, dt, ps=sf.STD_PRECISION):
    """Добавляет седловое с.р. и его сепаратрисы на фазовый портрет"""

    saddle_color = 'crimson'

    params_avg = (np.array(params1) + np.array(params2)) / 2
    sys_avg = so.FourBiharmonicPhaseOscillators(*params_avg)

    rhs = sys_avg.getReducedSystem
    jac = sys_avg.getReducedSystemJac

    # поиск точки расхождения траекторий
    min_len = min(trajs[0].shape[1], trajs[1].shape[1])
    divergence_point = None
    for i in range(min_len):
        dist = np.linalg.norm(trajs[0][:, i] - trajs[1][:, i])
        if dist > threshold:  # пороговое значение расхождения ЗАДАТЬ В АРГУМЕНТЕ ФУНКЦИИ
            divergence_point = (trajs[0][:, i] + trajs[1][:, i]) / 2
            break

    if divergence_point is None:
        print("Trajectories did not diverge significantly.")
        return

    eq_obj = find_equilibrium_by_guess(rhs, jac, divergence_point)
    if sf.isSaddle(eq_obj, ps):
        print(f"Saddle found at {eq_obj.coordinates}")
        ex, ey, ez = eq_obj.coordinates
        ax.scatter(ex, ey, ez, c=saddle_color, s=150, marker='X', zorder=10)

        if sf.is3DSaddleWith1dU(eq_obj, ps):  # если одномер неуст, строим сепаратрисы
            print("Tracing unstable separatrices...")
            saddle_init_pts = sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, sf.pickBothSeparatrices, ps)
            if saddle_init_pts:
                for init in saddle_init_pts:
                    init = list(init)
                    traj_saddle = compute_trajectory(init, params_avg, n, dt)
                    ax.plot(traj_saddle[0], traj_saddle[1], traj_saddle[2],
                            color=saddle_color, linewidth=8, alpha=0.25, zorder=1)
    elif eq_obj is None:
        print("Saddle not found near divergence point")
        return
    else:
        print("No saddle equilibrium found near divergence point")


def plot_attractors_plt(params_set, views, plot_placeholder, start_pt=0, n=50000, dt=0.01, directory="", point_name=""):
    opacity = 1.0
    trajs = get_trajectories(params_set, n, dt)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(tetraX, tetraY, tetraZ, color='black', alpha=opacity, linewidth=0.5)
    # ax.plot(frameX, frameY, frameZ, color='black', alpha=opacity, linewidth=0.5)
    # ax.plot(plankX, plankY, plankZ, color='black', alpha=opacity, linewidth=0.5)

    if len(trajs) == 1:
        traj = trajs[0].copy()
        point_colors = list(map(color_pt_by_domain, traj.T[start_pt:]))
        given_colors = ['red', 'blue', 'green', 'orange']
        for color in given_colors:
            color_group_idxs = [i for i, c in enumerate(point_colors) if c == color]
            for _, group in groupby(enumerate(color_group_idxs), key=lambda x: x[1] - x[0]):
                idxs = [item[1] for item in list(group)]
                ax.plot(traj[0][start_pt:][idxs], traj[1][start_pt:][idxs], traj[2][start_pt:][idxs], color=color, linewidth=1.5, zorder=2)
            if start_pt == 0:
                ax.scatter(traj[0][0], traj[1][0], traj[2][0], c='green', s=100, marker='D')
    else:
        cycle_colors = cycle(['orange', 'magenta', 'lime'])
        for i, traj in enumerate(trajs):
            if start_pt == 0:
                ax.scatter(traj[0][0], traj[1][0], traj[2][0], c='green', s=100, marker='D')
            ax.plot(traj[0][start_pt:], traj[1][start_pt:], traj[2][start_pt:],
                    color=next(cycle_colors), linewidth=(i+1)*1.5)

    if plot_placeholder is not None:
        plot_placeholder(ax, trajs)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True, alpha=0.3)

    for view_num, view in enumerate(views):
        ax.view_init(elev=view['elev'], azim=view['azim'], roll=view['roll'])
        ax.set_xlim3d(view['xlim_left'], view['xlim_right'])
        ax.set_ylim3d(view['ylim_left'], view['ylim_right'])
        ax.set_zlim3d(view['zlim_left'], view['zlim_right'])
        ax.set_axis_off()
        plt.tight_layout()

        assert directory != "" and point_name != "", "Enter directory name and point name"
        plt.savefig(f"{directory}/{point_name}_view_{view_num}.pdf", bbox_inches='tight')

    plt.close(fig)
