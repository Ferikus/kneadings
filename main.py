import numpy as np
from numba import cuda, njit
from mapping.convert import decimal_to_binary

DIM = 4
DIM_REDUCED = DIM - 1
THREADS_PER_BLOCK = 512

INFINITY = 10

InfinityError = -0.2
KneadingDoNotEndError = -0.1


@cuda.jit
def det4x4(m, det):
    det = 0.0
    sign = 1.0

    minor = [0.] * 9

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

        det += sign * m[0 * 4 + col] * det_minor
        sign *= -1.0


@cuda.jit
def bary_expansion(pt, bary_coords):
    """
    globalPtCoords must be a 3d vector with 0 <= x <= y <= z <= 2pi, i.e. inside a CIR
    returns an expansion of (globalPtCoords - center of mass) in barycentric coordinates
    """

    pt_o = cuda.local.array(3, dtype=np.float64)
    pt_a = cuda.local.array(3, dtype=np.float64)
    pt_b = cuda.local.array(3, dtype=np.float64)
    pt_c = cuda.local.array(3, dtype=np.float64)
    pt_w = cuda.local.array(3, dtype=np.float64)
    vec_wa = cuda.local.array(3, dtype=np.float64)
    vec_wb = cuda.local.array(3, dtype=np.float64)
    vec_wc = cuda.local.array(3, dtype=np.float64)
    vec_wo = cuda.local.array(3, dtype=np.float64)
    mat_bary = cuda.local.array(4 * 4, dtype=np.float64)
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

    main_det = 0
    det4x4(mat_bary, main_det)

    if abs(main_det) < 1e-12:
        bary_coords[:] = 0.  # сработает ли на cuda?
        return bary_coords  # ретурн точно нельзя -> либо в конец либо флаг

    # заполняем координаты решая систему методом Крамера
    for col in range(4):
        modified_mat = mat_bary.copy()  # прописать копирование вручную
        for row in range(4):
            modified_mat[4 * row + col] = rhs[row]

        coord_det = det4x4(modified_mat)
        bary_coords[col] = coord_det / main_det


@cuda.jit
def get_domain_num(bary_expansion, domain_num):
    min_coord = bary_expansion[0]
    i = 0
    while i < 4:
        if bary_expansion[domain_num] < min_coord:
            min_coord = bary_expansion[domain_num]
            domain_num = i


@cuda.jit
def full_rhs(params, phis, rhs_phis):
    """Calculates the right-hand side of the full system"""
    w, a, b, r = params
    rhs_phis[:] = w  # все 4 элемента
    for i in range(4):
        for j in range(4):
            rhs_phis[i] += 0.25 * (-np.sin(phis[i] - phis[j] + a) + r * np.sin(2 * (phis[i] - phis[j]) + b))


@cuda.jit
def reduced_rhs(params, psis):  # подогнать под cuda
    """Calculates the right-hand side of the reduced system"""
    phis = [0.] + psis
    rhs_phis = full_rhs(phis, params)
    rhs_psis = [0.] * 4
    for i in range(4):
        rhs_psis[i] = rhs_phis[i] - rhs_phis[0]
    return rhs_psis[1:]


# @cuda.jit
# def avg_face_distance(globalPt):
#     x, y, z = globalPt
#     return 0.25 * (x**2 + (y - x)**2 + (z - y)**2 + (z - 2 * np.pi)**2)


@cuda.jit
def avg_face_dist_deriv(params, pt, afdd):
    """Average distance from the point to the faces of the thetrahedron"""
    x, y, z = pt
    sys_curr = reduced_rhs(pt)
    afdd = (1.0*x - 0.5*y) * sys_curr[0] + (-0.5*x + 1.0*y - 0.5*z) * sys_curr[1] + (-0.5*y + 1.0*z - np.pi) * sys_curr[2]


@cuda.jit
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


@cuda.jit
def integrator_rk4(y_curr, params, dt, n, stride, kneadings_start, kneadings_end):
    """Calculates kneadings during integration"""
    # n -- количество шагов интегрирования
    # stride -- через сколько шагов начинаем считать нидинги
    # first_derivative_curr, prev -- значения производных системы на текущем шаге и на предыдущем

    bary_coords = cuda.local.array(DIM_REDUCED, dtype=np.float64)

    deriv_prev = 0
    deriv_curr = 0
    kneading_index = 0
    kneadings_weighted_sum = 0
    domain_num = 0

    avg_face_dist_deriv(params, y_curr, deriv_prev)

    for i in range(1, n):

        for j in range(stride):
            stepper_rk4(params, y_curr, dt)

        bary_expansion(y_curr, bary_coords)  # получаем барицентрические координаты точки
        get_domain_num(bary_coords, domain_num)  # получаем номер её подтетраэдра

        for k in range(DIM_REDUCED):
            if y_curr[k] > INFINITY or y_curr[k] < -INFINITY:
                return InfinityError

        avg_face_dist_deriv(params, y_curr, deriv_curr)

        # проверяем, происходит ли max по расстоянию
        if deriv_prev > 0 > deriv_curr:

            if kneading_index >= kneadings_start:
                kneadings_weighted_sum += domain_num * 1 / (4.0 ** (-kneading_index + kneadings_end + 1))
            kneading_index += 1

        deriv_prev = deriv_curr

        if kneading_index > kneadings_end:
            return kneadings_weighted_sum

    return KneadingDoNotEndError


@cuda.jit
def sweep_threads(
    kneadings_weighted_sum_set_gpu,
    y_inits,
    a_start,
    a_end,
    a_count,
    b_start,
    b_end,
    b_count,
    dt,
    n,
    stride,
    kneadings_start,
    kneadings_end,
):
    """CUDA kernel"""
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    a_step = (a_end - a_start) / (a_count - 1)
    b_step = (b_end - b_start) / (b_count - 1)

    if idx < a_count * b_count:

        i = idx // a_count
        j = idx % a_count

        params = cuda.local.array(2, dtype=np.float64)
        y_init = cuda.local.array(DIM_REDUCED, dtype=np.float64)

        params[0] = a_start + i * a_step
        params[1] = b_start + j * b_step

        y_init[0] = y_inits[idx * DIM_REDUCED + 0]
        y_init[1] = y_inits[idx * DIM_REDUCED + 1]
        y_init[2] = y_inits[idx * DIM_REDUCED + 2]

        kneadings_weighted_sum_set_gpu[i * b_count + j] = integrator_rk4(y_init, params, dt, n, stride,
                                                                         kneadings_start, kneadings_end)


def sweep(
    kneadings_weighted_sum_set,
    y_inits,
    a_start,
    a_end,
    a_count,
    b_start,
    b_end,
    b_count,
    dt,
    n,
    stride,
    kneadings_start,
    kneadings_end,
):
    """Calls CUDA kernel and gets kneadings set back from GPU"""
    total_parameter_space_size = a_count * b_count
    kneadings_weighted_sum_set_gpu = cuda.device_array(total_parameter_space_size)

    y_inits_gpu = cuda.device_array(len(y_inits))
    for i in range(len(y_inits)):
        y_inits_gpu[i] = y_inits[i]

    grid_x_dimension = (total_parameter_space_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    dim_grid = grid_x_dimension
    dim_block = THREADS_PER_BLOCK

    print(f"Num of blocks per grid:       {dim_grid}")
    print(f"Num of threads per block:     {dim_block}")
    print(f"Total Num of threads running: {dim_grid * dim_block}")
    print(f"Parameters aCount = {a_count}, bCount = {b_count}")

    # Call CUDA kernel
    sweep_threads[dim_grid, dim_block](  # blocks, threads
        kneadings_weighted_sum_set_gpu,
        y_inits_gpu,
        a_start,
        a_end,
        a_count,
        b_start,
        b_end,
        b_count,
        dt,
        n,
        stride,
        kneadings_start,
        kneadings_end,
    )

    kneadings_weighted_sum_set_gpu.copy_to_host(kneadings_weighted_sum_set)

    return kneadings_weighted_sum_set


if __name__ == "__main__":
    dt = 0.01
    n = 30000
    stride = 1
    max_kneadings = 7
    sweep_size = 300
    kneadings_weighted_sum_set = np.zeros(sweep_size * sweep_size)

    # сделать присваивание границ a и b + передача массива с нач коорд через файл npz из base_analysis

    a_start = 0.0
    a_end = 2.2
    b_start = 0.0
    b_end = 1.5

    inits_data = np.load(r'./inits.npz')

    inits = inits_data['inits']
    nones = inits_data['nones']

    # y_inits = [1e-8, 0.0, 0.0] * sweep_size * sweep_size
    # добавить проверку на размерность == DIM ?

    sweep(
        kneadings_weighted_sum_set,
        inits,
        a_start,
        a_end,
        sweep_size,
        b_start,
        b_end,
        sweep_size,
        dt,
        n,
        stride,
        0,
        max_kneadings
    )

    np.savez(
        'kneadings.npz',
        a_start=a_start,
        a_end=a_end,
        b_start=b_start,
        b_end=b_end,
        sweep_size=sweep_size,
        kneadings=kneadings_weighted_sum_set
    )

    print("Results:")
    for idx in range(sweep_size * sweep_size):
        i = idx // sweep_size
        j = idx % sweep_size

        kneading_weighted_sum = kneadings_weighted_sum_set[idx]
        kneading_symbolic = decimal_to_binary(kneading_weighted_sum)

        print(f"a: {a_start + i * (a_end - a_start) / (sweep_size - 1):.2f}, "
              f"b: {b_start + j * (b_end - b_start) / (sweep_size - 1):.2f} => "
              f"{kneading_symbolic} (Raw: {kneading_weighted_sum})")