import os
# needs to appear before `from numba import cuda`
os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
# set to "1" for more debugging, but slower performance
os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"

import numpy as np
from numba import cuda

from src.cuda_sweep.sweep_fbpo import stepper_rk4, reduced_rhs, event_cross_plane, bary_expansion, get_domain_num

DIM = 4
DIM_REDUCED = DIM - 1
PARAMS_COUNT = 4
MISC_DATA_BLOCK_LEN = DIM_REDUCED

THREADS_PER_BLOCK = 512

INFINITY = 10
MAX_KNEADINGS_LENGTH = 20_000
MAX_PERIOD_LENGTH = MAX_KNEADINGS_LENGTH // 2
ALPHABET_SIZE = 4

# CannotProcessSequenceError = -1
KneadingDoNotEndError = -0.1
InfinityError = -0.85
InEquilibriumError = -0.20
NoInitFoundError = -1.0

@cuda.jit(device=True)
def find_period_of_sequence(sequence, len_sequence):
    max_period = len_sequence // 2
    period_found = True
    for curr_period in range(1, max_period + 1):
        period_found = True
        for ind in range(curr_period, len_sequence):
            if sequence[ind] != sequence[ind - curr_period]:
                period_found = False
                break

        if period_found:
            break

    return curr_period if period_found else 0


@cuda.jit(device=True)
def find_complexity_of_sequence(sequence, len_sequence):
    period_charset = cuda.local.array(ALPHABET_SIZE, dtype=np.int8)
    complexity = 0

    for i in range(len_sequence):
        char_is_new = True
        for j in range(complexity):
            if period_charset[j] == sequence[i]:
                char_is_new = False
                break
        if char_is_new:
            period_charset[complexity] = sequence[i]
            complexity += 1

    return complexity


@cuda.jit(device=True)
def find_period_complexity_of_sequence(sequence, len_sequence):
    period = cuda.local.array(MAX_PERIOD_LENGTH, dtype=np.int8)

    max_period = len_sequence // 2
    period_found = True
    for curr_period in range(1, max_period + 1):
        period_found = True
        for ind in range(curr_period, len_sequence):
            # form the period symbol by symbol
            period[ind - curr_period] = sequence[ind - curr_period]
            if sequence[ind] != sequence[ind - curr_period]:
                period_found = False
                break

        if period_found:
            # calculate period complexity as array length now
            # complexity = 0
            # for i in range(curr_period):
            #     char_is_new = True
            #     for j in range(complexity):
            #         if period_charset[j] == period[i]:
            #             char_is_new = False
            #             break
            #     if char_is_new:
            #         period_charset[complexity] = period[i]
            #         complexity += 1
            complexity = find_complexity_of_sequence(period, curr_period)
            break

    return complexity if period_found else 0


def make_integrator_rk4_for_sequence(event_condition, process_sequence):
    @cuda.jit(device=True)
    def integrator_rk4_for_sequence(y_curr, params, dt, n, stride, kneadings_start, kneadings_end, misc):
        """Instead of calculating usual kneading weighted sum, returns a kneading sequence in array form"""
        y_prev = cuda.local.array(DIM_REDUCED, dtype=np.float64)
        rhs = cuda.local.array(DIM_REDUCED, dtype=np.float64)
        kneading_sequence = cuda.local.array(MAX_KNEADINGS_LENGTH, dtype=np.int8)
        sequence_len = kneadings_end - kneadings_start + 1

        for k in range(DIM_REDUCED):
            y_prev[k] = y_curr[k]

        kneading_index = 0

        for i in range(1, n):

            for j in range(stride):
                stepper_rk4(params, y_curr, dt)

            for j in range(DIM_REDUCED):
                if abs(y_curr[j]) > INFINITY:
                    return InfinityError  # CannotProcessSequenceError

            reduced_rhs(params, y_curr, rhs)
            if abs(rhs[0]) < 1e-8 and abs(rhs[1]) < 1e-8 and abs(rhs[2]) < 1e-8:
                return InEquilibriumError  # CannotProcessSequenceError

            if event_condition(params, y_prev, y_curr, misc):
                # write the symbol into the sequence array
                if kneading_index >= kneadings_start:
                    curr_bary = cuda.local.array(DIM, dtype=np.float64)
                    bary_expansion(y_curr, curr_bary)
                    curr_domain = get_domain_num(curr_bary)
                    kneading_sequence[kneading_index - kneadings_start] = curr_domain
                kneading_index += 1

            if kneading_index > kneadings_end:
                # the sequence is full, can process it now
                result = process_sequence(kneading_sequence, sequence_len)
                # print(kneading_sequence[0], kneading_sequence[1], kneading_sequence[2], kneading_sequence[3],
                #       kneading_sequence[4], kneading_sequence[5], kneading_sequence[6], kneading_sequence[7])
                # print(seq_period)
                return result

            for k in range(DIM_REDUCED):
                y_prev[k] = y_curr[k]

        return KneadingDoNotEndError  # CannotProcessSequenceError

    return integrator_rk4_for_sequence


def make_sweep_period_threads(event_condition, process_sequence):
    integrator_rk4_for_sequence = make_integrator_rk4_for_sequence(event_condition, process_sequence)
    @cuda.jit
    def sweep_period_threads(
            period_set,
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
            misc_set
    ):
        """CUDA kernel"""
        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        grid_count = (left_n + right_n + 1) * (up_n + down_n + 1)

        if idx < grid_count:

            is_in_nones = False
            for i in range(len(nones)):
                if idx == nones[i]:
                    is_in_nones = True
                    period_set[idx] = NoInitFoundError
                    break
            if is_in_nones == False:
                init = cuda.local.array(DIM_REDUCED, dtype=np.float64)
                params = cuda.local.array(PARAMS_COUNT, dtype=np.float64)
                misc = cuda.local.array(MISC_DATA_BLOCK_LEN, dtype=np.float64)

                for i in range(DIM_REDUCED):
                    init[i] = inits[idx * DIM_REDUCED + i]

                for i in range(PARAMS_COUNT):
                    params[i] = def_params[i]
                params[param_x_idx] = params_x[idx]
                params[param_y_idx] = params_y[idx]

                if len(misc_set) == MISC_DATA_BLOCK_LEN:
                    for i in range(MISC_DATA_BLOCK_LEN):
                        misc[i] = misc_set[i]
                else:
                    for i in range(MISC_DATA_BLOCK_LEN):
                        misc[i] = misc_set[idx * MISC_DATA_BLOCK_LEN + i]

                period_set[idx] = integrator_rk4_for_sequence(init, params, dt, n, stride, kneadings_start, kneadings_end, misc)

    return sweep_period_threads


def make_sweep_for_sequence(event_condition, process_sequence):
    def sweep_for_sequence(
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
            misc_set
    ):
        """Calls CUDA kernel and gets kneadings set back from GPU"""
        total_parameter_space_size = (left_n + right_n + 1) * (up_n + down_n + 1)
        assert len(misc_set) == MISC_DATA_BLOCK_LEN or total_parameter_space_size == len(misc_set) / MISC_DATA_BLOCK_LEN, \
            "Failed to unpack misc"

        result_set = np.zeros(total_parameter_space_size, dtype=np.float64)  # np.int8
        result_set_gpu = cuda.device_array(total_parameter_space_size, dtype=np.float64)  # np.int8

        inits_gpu = cuda.to_device(inits)
        nones_gpu = cuda.to_device(nones)
        def_params_gpu = cuda.to_device(def_params)
        params_x_gpu = cuda.to_device(params_x)
        params_y_gpu = cuda.to_device(params_y)
        misc_set_gpu = cuda.to_device(misc_set)

        param_x_idx = param_to_index[param_x_str]
        param_y_idx = param_to_index[param_y_str]

        dim_grid = (total_parameter_space_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        dim_block = THREADS_PER_BLOCK

        print(f"Num of blocks per grid:       {dim_grid}")
        print(f"Num of threads per block:     {dim_block}")
        print(f"Total Num of threads running: {dim_grid * dim_block}")
        print(f"Parameters a_count = {left_n + right_n + 1}, b_count = {up_n + down_n + 1}")

        # call CUDA kernel
        sweep_period_threads = make_sweep_period_threads(event_condition, process_sequence)
        sweep_period_threads[dim_grid, dim_block](  # blocks, threads
            result_set_gpu,
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
            misc_set_gpu
        )

        result_set_gpu.copy_to_host(result_set)

        return result_set

    return sweep_for_sequence


def sweep_regularity(inits, nones, params_x, params_y, def_params, param_to_index, param_x_str, param_y_str,
                            up_n, down_n, left_n, right_n, dt, n, stride, kneadings_start, kneadings_end, misc_set):
    # set up for integrator
    event_condition = event_cross_plane
    process_sequence = find_period_of_sequence
    sweep_for_sequence = make_sweep_for_sequence(event_condition, process_sequence)
    return sweep_for_sequence(inits, nones, params_x, params_y, def_params, param_to_index, param_x_str, param_y_str,
                              up_n, down_n, left_n, right_n, dt, n, stride, kneadings_start, kneadings_end, misc_set)


def sweep_complexity(inits, nones, params_x, params_y, def_params, param_to_index, param_x_str, param_y_str,
                     up_n, down_n, left_n, right_n, dt, n, stride, kneadings_start, kneadings_end, misc_set):
    # set up for integrator
    event_condition = event_cross_plane
    process_sequence = find_complexity_of_sequence
    sweep_for_sequence = make_sweep_for_sequence(event_condition, process_sequence)
    return sweep_for_sequence(inits, nones, params_x, params_y, def_params, param_to_index, param_x_str, param_y_str,
                              up_n, down_n, left_n, right_n, dt, n, stride, kneadings_start, kneadings_end, misc_set)


def sweep_period_complexity(inits, nones, params_x, params_y, def_params, param_to_index, param_x_str, param_y_str,
                            up_n, down_n, left_n, right_n, dt, n, stride, kneadings_start, kneadings_end, misc_set):
    # set up for integrator
    event_condition = event_cross_plane
    process_sequence = find_period_complexity_of_sequence
    sweep_for_sequence = make_sweep_for_sequence(event_condition, process_sequence)
    return sweep_for_sequence(inits, nones, params_x, params_y, def_params, param_to_index, param_x_str, param_y_str,
                              up_n, down_n, left_n, right_n, dt, n, stride, kneadings_start, kneadings_end, misc_set)
