import numpy as np
from src.system_analysis.taskutils import bary_expansion, get_domain_num, t

import lib.eq_finder.SystOsscills as so
from src.system_analysis.find_equilibrium import correct_equilibrium_coords, find_init_pts

DIM = 4
DIM_REDUCED = DIM - 1
INFINITY = 10

KneadingDoNotEndError = -0.1
InfinityError = -0.85
InEquilibriumError = -0.20
NoInitFoundError = -1.0


def full_rhs(params, phis):
    """Calculates the right-hand side of the full system"""
    w, a, b, r = params
    rhs_phis = [w] * 4
    for i in range(4):
        for j in range(4):
            rhs_phis[i] += 0.25 * (-np.sin(phis[i] - phis[j] + a) + r * np.sin(2 * (phis[i] - phis[j]) + b))
    return rhs_phis


def reduced_rhs(params, psis):
    """Calculates the right-hand side of the reduced system"""
    phis = [0.] + psis
    rhs_phis = full_rhs(params, phis)
    rhs_psis = [0.] * 4
    for i in range(4):
        rhs_psis[i] = rhs_phis[i] - rhs_phis[0]
    return rhs_psis[1:]


def avg_face_dist_deriv(params, pt):
    """Average distance from the point to the faces of the thetrahedron"""
    x, y, z = pt
    sys_curr = reduced_rhs(params, pt)
    afdd = (1.0*x - 0.5*y) * sys_curr[0] + (-0.5*x + 1.0*y - 0.5*z) * sys_curr[1] + (-0.5*y + 1.0*z - np.pi) * sys_curr[2]
    return afdd


def stepper_rk4(rhs, params, y_curr, dt):
    dim = len(y_curr)

    k1 = rhs(params, y_curr)

    y_temp = [y_curr[i] + k1[i] * dt / 2.0 for i in range(dim)]
    k2 = rhs(params, y_temp)

    y_temp = [y_curr[i] + k2[i] * dt / 2.0 for i in range(dim)]
    k3 = rhs(params, y_temp)

    y_temp = [y_curr[i] + k3[i] * dt for i in range(dim)]
    k4 = rhs(params, y_temp)

    return [y_curr[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6.0 for i in range(dim)]


def heavy_tail(state_curr, kneading_index, kneadings_end):
    curr_bary = bary_expansion(state_curr)
    curr_domain = get_domain_num(curr_bary)
    return curr_domain * 1 / (4.0 ** (-kneading_index + kneadings_end + 1))  # np.longdouble


def get_plane_coeffs(pt1, pt2, pt3):
    v1 = pt2 - pt1
    v2 = pt3 - pt2
    cp = np.cross(v1, v2)
    d = -np.dot(cp, pt1)
    return np.array([cp[0], cp[1], cp[2], d])


def plane_func(pt, plane_coeffs):
    return np.dot(plane_coeffs[:3], pt) + plane_coeffs[3]


def set_poincare_section_coeffs(domain_num, inner_sf):
    pt1 = np.array([0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi])  # pt_w

    if domain_num == 0:
        pt2 = np.array([1.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi])
    elif domain_num == 1:
        pt2 = np.array([1.0 * np.pi, 1.0 * np.pi, 1.0 * np.pi])
    elif domain_num == 2:
        pt2 = np.array([0.0, 0.0, 1.0 * np.pi])
    elif domain_num == 3:
        pt2 = np.array([0.0, 1.0 * np.pi, 2.0 * np.pi])

    inner_sf_temp = inner_sf.copy()
    for _ in range(domain_num):
        inner_sf_temp = t(inner_sf_temp)

    coeffs = get_plane_coeffs(pt1, pt2, inner_sf_temp)

    if domain_num == 0 or domain_num == 2:
        coeffs = -coeffs
    return coeffs


def event_cross_plane(state_prev, state_curr, inner_sf):
    prev_bary = bary_expansion(state_prev)
    prev_domain = get_domain_num(prev_bary)
    prev_coeffs = set_poincare_section_coeffs(prev_domain, inner_sf)
    prev_pl_val = plane_func(state_prev, prev_coeffs)

    curr_bary = bary_expansion(state_curr)
    curr_domain = get_domain_num(curr_bary)
    curr_coeffs = set_poincare_section_coeffs(curr_domain, inner_sf)
    curr_pl_val = plane_func(state_curr, curr_coeffs)

    if prev_domain == curr_domain:
        if prev_pl_val < 0 < curr_pl_val:
            return True
    return False


def make_integrator_rk4(event_condition, kneading_evaluator):
    def integrator_rk4(y_curr, params, dt, n, stride, kneadings_start, kneadings_end, inner_sf, debug=True):
        y_prev = y_curr.copy()
        kneading_index = 0
        kneadings_weighted_sum = 0

        trajectory = np.zeros((n, DIM_REDUCED))
        trajectory[0] = y_curr.copy()
        extrs = []
        ns = []
        last_n = 0

        for i in range(1, n):
            for j in range(stride):
                y_curr = stepper_rk4(reduced_rhs, params, y_curr, dt)
            trajectory[i] = y_curr.copy()

            infinity_flag = 0
            for k in range(DIM_REDUCED):
                if y_curr[k] > INFINITY or y_curr[k] < -INFINITY:
                    infinity_flag = 1
            if infinity_flag:
                break

            # print(abs(y_curr[0]), abs(y_curr[1]), abs(y_curr[2]))
            if abs(y_curr[0]) < 1e-8 and abs(y_curr[1]) < 1e-8 and abs(y_curr[2]) < 1e-8:
                if debug: print("НЕДОСЧЁТ")
                break

            if event_condition(y_prev, y_curr, inner_sf):
                if kneading_index >= kneadings_start:
                    kneadings_weighted_sum += kneading_evaluator(y_curr, kneading_index, kneadings_end)

                    if debug:
                        curr_bary = bary_expansion(y_curr)
                        curr_domain = get_domain_num(curr_bary)
                        print(curr_domain)

                kneading_index += 1
                extrs.append(y_curr.copy())
                ns.append(i)
                if debug: print("ПОВЫСИЛИ ИНДЕКС")

            last_n = i
            if kneading_index > kneadings_end:
                break

            y_prev = y_curr.copy()

        if debug: print("КОНЕЦ")
        return kneadings_weighted_sum, trajectory[:last_n], ns, np.array(extrs), last_n

    return integrator_rk4


# def sweep(
#         inits, nones, params_x, params_y, def_params,
#         param_x_idx, param_y_idx,
#         dt, n, stride, kneadings_start, kneadings_end,
#         inner_sf_set
# ):
#     total_size = len(params_x)
#     results = np.zeros(total_size)
#
#     get_kneading = make_get_kneading_generalized(event_cross_plane, heavy_tail)
#
#     for idx in range(total_size):
#         if idx in nones:
#             results[idx] = NoInitFoundError
#             continue
#
#         current_params = def_params.copy()
#         current_params[param_x_idx] = params_x[idx]
#         current_params[param_y_idx] = params_y[idx]
#
#         init_point = inits[idx * DIM_REDUCED: (idx + 1) * DIM_REDUCED]
#
#         if len(inner_sf_set) == DIM_REDUCED:
#             current_inner_sf = inner_sf_set
#         else:
#             current_inner_sf = inner_sf_set[idx * DIM_REDUCED: (idx + 1) * DIM_REDUCED]
#
#         res_tuple = get_kneading(
#             init_point, current_params, dt, n, stride,
#             kneadings_start, kneadings_end, current_inner_sf
#         )
#
#         results[idx] = res_tuple[0]
#
#     return results


if __name__ == "__main__":
    # a: -2.878590800000000, b: -1.678849700000000 => 1000000000 (Raw: 9.5367431640625e-07)

    params = [0.0, -2.8785920, -1.6788497, 1.0]
    sys = so.FourBiharmonicPhaseOscillators(*params)
    reduced_rhs_wrapper = sys.getReducedSystem
    reduced_jac_wrapper = sys.getReducedSystemJac

    # start_eq = [0.0, 2.30956058, 4.75652024]
    inner_sf = [1.427257804280822, 3.2091500304528755, 4.414529919493724]

    # start_eq = correct_equilibrium_coords(reduced_rhs, reduced_jac, start_eq)
    inner_sf = correct_equilibrium_coords(reduced_rhs_wrapper, reduced_jac_wrapper, inner_sf)
    y_curr = list(find_init_pts(sys))
    print(y_curr, inner_sf)

    event_condition = event_cross_plane
    kneading_evaluator = heavy_tail
    integrator_rk4 = make_integrator_rk4(event_condition, kneading_evaluator)
    kneadings_weighted_sum, trajectory, ns, extrs, last_n = integrator_rk4(y_curr, params, dt=0.01, n=300000, stride=1,
                                                                           kneadings_start=0, kneadings_end=20,
                                                                           inner_sf=inner_sf)