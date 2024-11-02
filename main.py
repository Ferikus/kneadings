import numpy as np
from numba import cuda, njit

DIM = 3
THREADS_PER_BLOCK = 512
INFINITY = 100000

InfinityError = -0.2
KneadingDoNotEndError = -0.1


# Шаг РК-4, сохраняет значение в y_curr
@cuda.jit
def stepper_rk4(params, y_curr, dt):
    k1 = cuda.local.array(DIM, dtype=np.float32)
    k2 = cuda.local.array(DIM, dtype=np.float32)
    k3 = cuda.local.array(DIM, dtype=np.float32)
    k4 = cuda.local.array(DIM, dtype=np.float32)
    func = cuda.local.array(DIM, dtype=np.float32)

    rhs(params, y_curr, k1)

    for i in range(DIM):
        func[i] = y_curr[i] + k1[i] / 2.0
    rhs(params, func, k2)

    for i in range(DIM):
        func[i] = y_curr[i] + k2[i] / 2.0
    rhs(params, func, k3)

    for i in range(DIM):
        func[i] = y_curr[i] + k3[i]
    rhs(params, func, k4)

    for i in range(DIM):
        y_curr[i] = y_curr[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6.0


# Значение системы ДУ в точке
@cuda.jit
def rhs(params, y, dydt):
    a, b = params
    dydt[0] = y[1]
    dydt[1] = y[2]
    dydt[2] = -b * y[2] - y[1] + a * y[0] - a * y[0] ** 3


@cuda.jit
def integrator_rk4(y_curr, params, dt, n, stride, kneadings_start, kneadings_end):
    # n -- количество шагов интегрирования
    # stride -- через сколько шагов начинаем считать нидинги
    # first_derivative_curr, prev -- значения производных системы на текущем шаге и на предыдущем

    first_derivative_prev = cuda.local.array(DIM, dtype=np.float32)
    first_derivative_curr = cuda.local.array(DIM, dtype=np.float32)
    kneading_index = 0
    kneadings_weighted_sum = 0

    rhs(params, y_curr, first_derivative_prev)

    for i in range(n):

        j = 0
        while j < stride:
            stepper_rk4(params, y_curr, dt)
            j += 1
        for k in range(DIM):
            if y_curr[k] > INFINITY or y_curr[k] < -INFINITY:
                return InfinityError

        rhs(params, y_curr, first_derivative_curr)
        if first_derivative_prev[0] * first_derivative_curr[0] < 0:

            if first_derivative_curr[1] < 0 and y_curr[0] > 1:
                if kneading_index >= kneadings_start:
                    # 1
                    kneadings_weighted_sum += 1 / (2.0 ** (-kneading_index + kneadings_end + 1))
                kneading_index += 1

            elif first_derivative_curr[1] > 0 and y_curr[0] < -1:
                # 0
                kneading_index += 1

        first_derivative_prev[0] = first_derivative_curr[0]

    if kneading_index > kneadings_end:
        return kneadings_weighted_sum
    # return kneading_index

    return KneadingDoNotEndError


# CUDA Kernel
@cuda.jit
def sweep_threads(
    kneadings_weighted_sum_gpu,
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
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    a_step = (a_end - a_start) / (a_count - 1)
    b_step = (b_end - b_start) / (b_count - 1)

    params = cuda.local.array(2, dtype=np.float32)
    y_init = cuda.local.array(DIM, dtype=np.float32)

    if idx < a_count * b_count:

        i = idx // a_count
        j = idx % a_count

        params[0] = a_start + i * a_step
        params[1] = b_start + j * b_step

        y_init[0], y_init[1], y_init[2] = 1e-8, 0.0, 0.0

        kneadings_weighted_sum_gpu[i * b_count + j] = integrator_rk4(y_init, params, dt, n, stride, kneadings_start, kneadings_end)


def sweep(
    kneadings_weighted_sum_set,
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

    total_parameter_space_size = a_count * b_count
    kneadings_weighted_sum_set_gpu = cuda.device_array(total_parameter_space_size)

    # Timing
    # start_event = cp.cuda.Event()
    # stop_event = cp.cuda.Event()
    #
    # start_event.record()

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

    # cp.cuda.Stream.null.synchronize()  # Synchronize to ensure all kernels have completed
    # stop_event.record()
    # stop_event.synchronize()

    # time_kernel = (
    #     cp.cuda.get_elapsed_time(start_event, stop_event) / 1000.0
    # )  # Convert to seconds
    # print(f"Total time (sec): {time_kernel}")

    kneadings_weighted_sum_set_gpu.copy_to_host(kneadings_weighted_sum_set)


if __name__ == "__main__":
    dt = 0.01
    n = 30000
    stride = 1
    max_kneadings = 20
    sweep_size = 5
    kneadings_weighted_sum_set = np.zeros(sweep_size * sweep_size)

    sweep(
        kneadings_weighted_sum_set,
        0.0,
        2.2,
        sweep_size,
        0,
        1.5,
        sweep_size,
        dt,
        n,
        stride,
        0,
        10,
    )

    for i in range(sweep_size):
        for j in range(sweep_size):
            print(kneadings_weighted_sum_set[i * sweep_size + j])
