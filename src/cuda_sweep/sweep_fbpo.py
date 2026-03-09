import numpy as np
from numba import cuda
from src.plotting.convert import convert_heavy_tail_to_sequence

PARAM_TO_INDEX = {
    'w': 0,
    'a': 1,
    'b': 2,
    'r': 3
}

DIM = 4
DIM_REDUCED = DIM - 1
PARAMS_COUNT = 4
THREADS_PER_BLOCK = 512

INFINITY = 10

KneadingDoNotEndError = -0.1
InfinityError = -0.20
InEquilibriumError = -0.21
NoInitFoundError = -0.3


##### System #####

@cuda.jit(device=True)
def full_rhs(params, phis, dphis):
    """Calculates the right-hand side of the full system"""
    w, a, b, r = params
    for i in range(4):
        dphis[i] = w
        for j in range(4):
            dphis[i] += 0.25 * (-np.sin(phis[i] - phis[j] + a) + r * np.sin(2 * (phis[i] - phis[j]) + b))


@cuda.jit(device=True)
def reduced_rhs(params, psis, dpsis):
    """Calculates the right-hand side of the reduced system"""
    phis = cuda.local.array(DIM, dtype=np.float64)
    dphis = cuda.local.array(DIM, dtype=np.float64)
    dpsis_temp = cuda.local.array(DIM, dtype=np.float64)

    phis[0] = 0.
    for i in range(3):
        phis[i + 1] = psis[i]

    full_rhs(params, phis, dphis)

    for i in range(DIM):
        dpsis_temp[i] = dphis[i] - dphis[0]
    for i in range(DIM_REDUCED):
        dpsis[i] = dpsis_temp[i + 1]


@cuda.jit(device=True)
def avg_face_dist_deriv(params, pt):
    """Average distance from the point to the faces of the thetrahedron"""
    x, y, z = pt
    sys_curr = cuda.local.array(DIM_REDUCED, dtype=np.float64)
    reduced_rhs(params, pt, sys_curr)
    afdd = ((1.0 * x - 0.5 * y) * sys_curr[0] + (-0.5 * x + 1.0 * y - 0.5 * z) * sys_curr[1]
            + (-0.5 * y + 1.0 * z - np.pi) * sys_curr[2])
    return afdd


@cuda.jit(device=True)
def stepper_rk4(params, y_curr, dt):
    """Makes RK-4 step and saves the value in y_curr"""
    k1 = cuda.local.array(DIM_REDUCED, dtype=np.float64)
    k2 = cuda.local.array(DIM_REDUCED, dtype=np.float64)
    k3 = cuda.local.array(DIM_REDUCED, dtype=np.float64)
    k4 = cuda.local.array(DIM_REDUCED, dtype=np.float64)
    y_temp = cuda.local.array(DIM_REDUCED, dtype=np.float64)

    reduced_rhs(params, y_curr, k1)

    for i in range(DIM_REDUCED):
        y_temp[i] = y_curr[i] + k1[i] * dt / 2.0
    reduced_rhs(params, y_temp, k2)

    for i in range(DIM_REDUCED):
        y_temp[i] = y_curr[i] + k2[i] * dt / 2.0
    reduced_rhs(params, y_temp, k3)

    for i in range(DIM_REDUCED):
        y_temp[i] = y_curr[i] + k3[i] * dt
    reduced_rhs(params, y_temp, k4)

    for i in range(DIM_REDUCED):
        y_curr[i] = y_curr[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6.0


##### Domain number #####

@cuda.jit(device=True)
def det4x4(m, det):
    det[0] = 0.0
    sign = 1.0

    minor = cuda.local.array(9, dtype=np.float64)
    for i in range(9):
        minor[i] = 0.0

    for col in range(4):
        minor_row_idx = 0
        for i in range(1, 4):
            minor_col_idx = 0
            for j in range(4):
                if j != col:
                    minor[minor_row_idx * 3 + minor_col_idx] = m[i * 4 + j]
                    minor_col_idx += 1
            minor_row_idx += 1

        det_minor = (
                minor[0] * (minor[4] * minor[8] - minor[5] * minor[7]) -
                minor[1] * (minor[3] * minor[8] - minor[5] * minor[6]) +
                minor[2] * (minor[3] * minor[7] - minor[4] * minor[6])
        )

        det[0] += sign * m[0 * 4 + col] * det_minor
        sign *= -1.0


@cuda.jit(device=True)
def bary_expansion(pt, bary_coords):
    pt_o = cuda.local.array(3, dtype=np.float64)
    pt_a = cuda.local.array(3, dtype=np.float64)
    pt_b = cuda.local.array(3, dtype=np.float64)
    pt_c = cuda.local.array(3, dtype=np.float64)
    pt_w = cuda.local.array(3, dtype=np.float64)
    vec_wa = cuda.local.array(3, dtype=np.float64)
    vec_wb = cuda.local.array(3, dtype=np.float64)
    vec_wc = cuda.local.array(3, dtype=np.float64)
    vec_wo = cuda.local.array(3, dtype=np.float64)
    mat_bary = cuda.local.array(16, dtype=np.float64)
    modified_mat = cuda.local.array(16, dtype=np.float64)
    rhs = cuda.local.array(4, dtype=np.float64)

    pt_o[0] = 0.0; pt_o[1] = 0.0; pt_o[2] = 0.0
    pt_a[0] = 0.0; pt_a[1] = 0.0; pt_a[2] = 2*np.pi
    pt_b[0] = 0.0; pt_b[1] = 2*np.pi; pt_b[2] = 2*np.pi
    pt_c[0] = 2*np.pi; pt_c[1] = 2*np.pi; pt_c[2] = 2*np.pi

    pt_w[0] = 0.25 * (pt_a[0] + pt_b[0] + pt_c[0] - 3 * pt_o[0])
    pt_w[1] = 0.25 * (pt_a[1] + pt_b[1] + pt_c[1] - 3 * pt_o[1])
    pt_w[2] = 0.25 * (pt_a[2] + pt_b[2] + pt_c[2] - 3 * pt_o[2])

    for i in range(3):
        vec_wa[i] = pt_a[i] - pt_w[i]
        vec_wb[i] = pt_b[i] - pt_w[i]
        vec_wc[i] = pt_c[i] - pt_w[i]
        vec_wo[i] = pt_o[i] - pt_w[i]

        mat_bary[4 * i] = vec_wa[i]
        mat_bary[4 * i + 1] = vec_wb[i]
        mat_bary[4 * i + 2] = vec_wc[i]
        mat_bary[4 * i + 3] = vec_wo[i]

        rhs[i] = pt[i] - pt_w[i]

    mat_bary[12] = 1.; mat_bary[13] = 1.; mat_bary[14] = 1.; mat_bary[15] = 1.
    rhs[3] = 1.

    main_det = cuda.local.array(1, dtype=np.float64)
    det4x4(mat_bary, main_det)

    if abs(main_det[0]) < 1e-12:
        for i in range(4):
            bary_coords[i] = 0.0
        return

    for col in range(4):
        for i in range(16):
            modified_mat[i] = mat_bary[i]
        for row in range(4):
            modified_mat[4 * row + col] = rhs[row]

        coord_det = cuda.local.array(1, dtype=np.float64)
        det4x4(modified_mat, coord_det)
        bary_coords[col] = coord_det[0] / main_det[0]


@cuda.jit(device=True)
def get_domain_num(bary_expansion):
    min_coord = bary_expansion[0]
    i = 0
    domain_num = i
    while i < 4:
        if bary_expansion[i] < min_coord:
            min_coord = bary_expansion[i]
            domain_num = i
        i += 1
    return domain_num


##### Poincare section #####

# @cuda.jit(device=True)
# def t(pt):
#     x, y, z = pt
#
#     x_new = y - x
#     y_new = z - x
#     z_new = 2 * np.pi - x
#
#     pt[0] = x_new
#     pt[1] = y_new
#     pt[2] = z_new
#
#
# @cuda.jit(device=True)
# def get_plane_coeffs(pt1, pt2, pt3, coeffs):
#     v1 = cuda.local.array(3, dtype=np.float64)
#     v2 = cuda.local.array(3, dtype=np.float64)
#
#     for i in range(3):
#         v1[i] = pt2[i] - pt1[i]
#         v2[i] = pt3[i] - pt2[i]
#
#     x1, y1, z1 = v1
#     x2, y2, z2 = v2
#
#     coeffs[0] = y1 * z2 - z1 * y2  # a
#     coeffs[1] = -(x1 * z2 - z1 * x2)  # b
#     coeffs[2] = x1 * y2 - y1 * x2  # c
#     coeffs[3] = -coeffs[0] * pt1[0] - coeffs[1] * pt1[1] - coeffs[2] * pt1[2]  # d
#
#
# @cuda.jit(device=True)
# def get_poincare_section_coeffs(domain_num, coeffs, inner_sf):
#     pt1 = cuda.local.array(3, dtype=np.float64)  # pt_w
#     pt2 = cuda.local.array(3, dtype=np.float64)  # (pt_b + pt_c) / 2
#
#     pt1[0] = 0.5*np.pi; pt1[1] = 1.0*np.pi; pt1[2] = 1.5*np.pi
#
#     if domain_num == 0:
#         pt2[0] = 1.0*np.pi; pt2[1] = 2.0*np.pi; pt2[2] = 2.0*np.pi
#     elif domain_num == 1:
#         pt2[0] = 1.0*np.pi; pt2[1] = 1.0*np.pi; pt2[2] = 1.0*np.pi
#     elif domain_num == 2:
#         pt2[0] = 0.0; pt2[1] = 0.0; pt2[2] = 1.0*np.pi
#     elif domain_num == 3:
#         pt2[0] = 0.0; pt2[1] = 1.0*np.pi; pt2[2] = 2.0*np.pi
#
#     for i in range(domain_num):  # добавить проверку, что inner_sf находится в 0 подтетраэдре
#         t(inner_sf)
#
#     get_plane_coeffs(pt1, pt2, inner_sf, coeffs)


@cuda.jit(device=True)
def plane_func(pt, plane_coeffs):
    x, y, z = pt
    a, b, c, d = plane_coeffs
    return a*x + b*y + c*z + d


@cuda.jit(device=True)
def set_poincare_section_coeffs(domain_num, coeffs_set, coeffs):
    for i in range(4):
        coeffs[i] = coeffs_set[domain_num * 4 + i]


###### Sweep components: events, evaluator, integrator, sweep ######

@cuda.jit(device=True)
def event_cross_plane(params, state_prev, state_curr, coeffs_set):
    bary_prev = cuda.local.array(DIM, dtype=np.float64)
    bary_curr = cuda.local.array(DIM, dtype=np.float64)
    coeffs_prev = cuda.local.array(DIM, dtype=np.float64)
    coeffs_curr = cuda.local.array(DIM, dtype=np.float64)

    bary_expansion(state_prev, bary_prev)
    domain_prev = get_domain_num(bary_prev)
    set_poincare_section_coeffs(domain_prev, coeffs_set, coeffs_prev)
    plane_val_prev = plane_func(state_prev, coeffs_prev)

    bary_expansion(state_curr, bary_curr)
    domain_curr = get_domain_num(bary_curr)
    set_poincare_section_coeffs(domain_curr, coeffs_set, coeffs_curr)
    plane_val_curr = plane_func(state_curr, coeffs_curr)

    evt_happened = False
    if domain_prev == domain_curr:
        if plane_val_prev < 0 < plane_val_curr:
            evt_happened = True

    return evt_happened


@cuda.jit(device=True)
def event_avg_face_dist_deriv_max(params, state_prev, state_curr):
    deriv_prev = avg_face_dist_deriv(params, state_prev)
    deriv_curr = avg_face_dist_deriv(params, state_curr)
    return deriv_curr < 0 < deriv_prev


@cuda.jit(device=True)
def kneading_evaluator(state_curr, kneading_index, kneadings_end, kneadings_weighted_sum):
    curr_bary = cuda.local.array(DIM, dtype=np.float64)
    bary_expansion(state_curr, curr_bary)
    curr_domain = get_domain_num(curr_bary)
    return kneadings_weighted_sum + curr_domain * 1 / (4.0 ** (-kneading_index + kneadings_end + 1))


def make_integrator_rk4(event_condition, kneading_encoder):
    @cuda.jit(device=True)
    def integrator_rk4(y_curr, params, dt, n, stride, kneadings_start, kneadings_end, misc):
        """Calculates kneadings during integration"""
        y_prev = cuda.local.array(DIM_REDUCED, dtype=np.float64)
        rhs = cuda.local.array(DIM_REDUCED, dtype=np.float64)

        for k in range(DIM_REDUCED):
            y_prev[k] = y_curr[k]

        kneading_index = 0
        kneadings_weighted_sum = 0

        for i in range(1, n):

            for j in range(stride):
                stepper_rk4(params, y_curr, dt)

            for k in range(DIM_REDUCED):
                if y_curr[k] > INFINITY or y_curr[k] < -INFINITY:
                    return InfinityError

            # reduced_rhs(params, y_curr, rhs)
            # if abs(rhs[0]) < 1e-6 and abs(rhs[1]) < 1e-6 and abs(rhs[2]) < 1e-6:
            #     return InEquilibriumError

            if event_condition(params, y_prev, y_curr, misc):

                if kneading_index >= kneadings_start:
                    kneadings_weighted_sum = kneading_encoder(y_curr, kneading_index, kneadings_end, kneadings_weighted_sum)

                kneading_index += 1

            if kneading_index > kneadings_end:
                return kneadings_weighted_sum

            for k in range(DIM_REDUCED):
                y_prev[k] = y_curr[k]

        return KneadingDoNotEndError

    return integrator_rk4


def make_sweep_threads(event_condition, kneading_encoder):
    integrator_rk4 = make_integrator_rk4(event_condition, kneading_encoder)
    @cuda.jit
    def sweep_threads(
            kneadings_weighted_sum_set,
            inits,
            nones,
            params_x,
            params_y,
            def_params,
            param_x_idx,
            param_y_idx,
            up_n,
            down_n,
            left_n,
            right_n,
            dt,
            n,
            stride,
            kneadings_start,
            kneadings_end,
            misc
    ):
        """CUDA kernel"""
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        grid_count = (left_n + right_n + 1) * (up_n + down_n + 1)

        if idx < grid_count:

            is_in_nones = False
            for i in range(len(nones)):
                if idx == nones[i]:
                    is_in_nones = True
                    kneadings_weighted_sum_set[idx] = NoInitFoundError
                    break
            if is_in_nones == False:
                init = cuda.local.array(DIM_REDUCED, dtype=np.float64)
                params = cuda.local.array(4, dtype=np.float64)

                for i in range(DIM_REDUCED):
                    init[i] = inits[idx * DIM_REDUCED + i]

                for i in range(PARAMS_COUNT):
                    params[i] = def_params[i]
                params[param_x_idx] = params_x[idx]
                params[param_y_idx] = params_y[idx]

                kneadings_weighted_sum_set[idx] = integrator_rk4(init, params, dt, n, stride, kneadings_start,
                                                                 kneadings_end, misc)

    return sweep_threads


###### General sweep function ######

def sweep(
        inits,
        nones,
        params_x,
        params_y,
        def_params,
        param_to_index,
        param_x_str,
        param_y_str,
        up_n,
        down_n,
        left_n,
        right_n,
        dt,
        n,
        stride,
        kneadings_start,
        kneadings_end,
        misc
):
    """Calls CUDA kernel and gets kneadings set back from GPU"""
    total_parameter_space_size = (left_n + right_n + 1) * (up_n + down_n + 1)
    kneadings_weighted_sum_set = np.zeros(total_parameter_space_size)
    kneadings_weighted_sum_set_gpu = cuda.device_array(total_parameter_space_size)

    inits_gpu = cuda.to_device(inits)
    nones_gpu = cuda.to_device(nones)
    def_params_gpu = cuda.to_device(def_params)
    params_x_gpu = cuda.to_device(params_x)
    params_y_gpu = cuda.to_device(params_y)
    misc_gpu = cuda.to_device(misc)

    param_x_idx = param_to_index[param_x_str]
    param_y_idx = param_to_index[param_y_str]

    grid_x_dimension = (total_parameter_space_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    dim_grid = grid_x_dimension
    dim_block = THREADS_PER_BLOCK

    print(f"Num of blocks per grid:       {dim_grid}")
    print(f"Num of threads per block:     {dim_block}")
    print(f"Total Num of threads running: {dim_grid * dim_block}")
    print(f"Parameters a_count = {left_n + right_n + 1}, b_count = {up_n + down_n + 1}")

    # set up for integrator

    # set 1
    # event_condition = event_avg_face_dist_deriv_max
    # kneading_encoder = kneading_evaluator

    # set 2
    event_condition = event_cross_plane
    kneading_encoder = kneading_evaluator

    # call CUDA kernel
    sweep_threads = make_sweep_threads(event_condition, kneading_encoder)
    sweep_threads[dim_grid, dim_block](  # blocks, threads
        kneadings_weighted_sum_set_gpu,
        inits_gpu,
        nones_gpu,
        params_x_gpu,
        params_y_gpu,
        def_params_gpu,
        param_x_idx,
        param_y_idx,
        up_n,
        down_n,
        left_n,
        right_n,
        dt,
        n,
        stride,
        kneadings_start,
        kneadings_end,
        misc_gpu
    )

    kneadings_weighted_sum_set_gpu.copy_to_host(kneadings_weighted_sum_set)

    return kneadings_weighted_sum_set


if __name__ == "__main__":
    dt = 0.01
    n = 50000
    stride = 1
    max_kneadings = 7

    inits_data = np.load(r'../system_analysis/inits1.npz')

    inits = inits_data['inits']
    nones = inits_data['nones']
    params_x = inits_data['alphas']
    params_y = inits_data['betas']
    up_n = int(inits_data['up_n'])
    down_n = int(inits_data['down_n'])
    left_n = int(inits_data['left_n'])
    right_n = int(inits_data['right_n'])

    # default parameter values
    w = 0.0
    a = -2.907273192326542
    b = -1.623684228842761
    r = 1.0
    def_params = [w, a, b, r]

    kneadings_weighted_sum_set = sweep(
        inits,
        nones,
        params_x,
        params_y,
        def_params,
        PARAM_TO_INDEX,
        'a',
        'b',
        up_n,
        down_n,
        left_n,
        right_n,
        dt,
        n,
        stride,
        0,
        max_kneadings
    )

    np.savez(
        'sweep_fbpo.npz',
        kneadings=kneadings_weighted_sum_set
    )

    print("Results:")
    for idx in range((left_n + right_n + 1) * (up_n + down_n + 1)):
        kneading_weighted_sum = kneadings_weighted_sum_set[idx]
        kneading_symbolic = convert_heavy_tail_to_sequence(kneading_weighted_sum, 4, max_kneadings)

        print(f"a: {params_x[idx]:.9f}, "
              f"b: {params_y[idx]:.9f} => "
              f"{kneading_symbolic} (Raw: {kneading_weighted_sum})")