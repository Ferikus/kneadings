import numpy as np
from numba import cuda
from src.cuda_sweep.sweep_fbpo import full_rhs
import itertools

DIM = 4
DIM_REDUCED = 3
THREADS_PER_BLOCK = 256
DETECTIVE_ENTRY_SIZE = 4
PAR_SPACE_DIM = 4

NO_INIT_FOUND_ERROR = -2.0
STUCK_INTO_EQUILIBRIUM = -1.0 
SUCCESSFULLY_COMPUTED = 1.0

CODE_T1_SYMMETRY = 1.0
CODE_T2_SYMMETRY = 2.0 
CODE_T0_SYMMETRY = 3.0

N_ALL_PERMUTATIONS = 24
N_CIR_PERMUTATIONS = 4

DATA_TO_COLOR = {
    NO_INIT_FOUND_ERROR: 'white',
    STUCK_INTO_EQUILIBRIUM: 'darkgray',
    CODE_T1_SYMMETRY: 'red',
    CODE_T2_SYMMETRY: 'yellow',
    CODE_T0_SYMMETRY: 'green'
}

@cuda.jit(device=True)
def stepper_rk4_full(params, y_curr, dt):
    """Makes RK-4 step and saves the value in y_curr"""
    k1 = cuda.local.array(DIM, dtype=np.float64)
    k2 = cuda.local.array(DIM, dtype=np.float64)
    k3 = cuda.local.array(DIM, dtype=np.float64)
    k4 = cuda.local.array(DIM, dtype=np.float64)
    y_temp = cuda.local.array(DIM, dtype=np.float64)

    full_rhs(params, y_curr, k1)

    for i in range(DIM):
        y_temp[i] = y_curr[i] + k1[i] * dt / 2.0
    full_rhs(params, y_temp, k2)

    for i in range(DIM):
        y_temp[i] = y_curr[i] + k2[i] * dt / 2.0
    full_rhs(params, y_temp, k3)

    for i in range(DIM):
        y_temp[i] = y_curr[i] + k3[i] * dt
    full_rhs(params, y_temp, k4)

    for i in range(DIM):
        y_curr[i] = y_curr[i] + (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6.0

@cuda.jit(device=True)
def ashwinDetective(state):
    x1 = state[0]
    x2 = state[1]
    x3 = state[2]
    return np.sin(3*x1)*np.sin(6*x2)*np.sin(9*x3)

@cuda.jit(device=True)
def trapezoidRule(prevState, curState):
    left = ashwinDetective(prevState)
    right = ashwinDetective(curState)
    return left + right

@cuda.jit(device=True)
def permuteStateInto(origState, permut, permutedState):
    for ind in range(DIM):
        permutedState[ind] = origState[permut[ind]]


@cuda.jit(device=True)
def symmetry_type_compute(initFullSys, params, dt, nStepsSkip, nStepsAttractor, out_array, fullPermutationsFlat, cirPermutationsFlat):
    curState = cuda.local.array(DIM, dtype=np.float64)
    prevState = cuda.local.array(DIM, dtype=np.float64)
    startCurState = cuda.local.array(DIM, dtype=np.float64)
    startPrevState = cuda.local.array(DIM, dtype=np.float64)
    permutedCurState = cuda.local.array(DIM, dtype=np.float64)
    permutedPrevState = cuda.local.array(DIM, dtype=np.float64)
    curRhs = cuda.local.array(DIM, dtype=np.float64)

    curRowPermutation = cuda.local.array(DIM, dtype=np.int16)
    curColPermutation = cuda.local.array(DIM, dtype=np.int16)

    notInEquilibrium = True 

    for i in range(DIM):
        curState[i] = initFullSys[i]

    # first we compute the transient before we get onto an attractor
    iterIndex = 0
    while (iterIndex <= nStepsSkip) and notInEquilibrium:
        # here was wrong orders of parameters passed
        stepper_rk4_full(params, curState, dt)
        iterIndex += 1

        # this is a check for getting stuck into equilibrium
        full_rhs(params, curState, curRhs)
        if abs(curRhs[0]) < 1e-8 and abs(curRhs[1]) < 1e-8 and abs(curRhs[2]) < 1e-8 and abs(curRhs[3]) < 1e-8:
            notInEquilibrium = False 


    # and now we compute on the attractor 
    iterIndex = 0
    kAs = cuda.local.array((N_ALL_PERMUTATIONS, N_CIR_PERMUTATIONS), dtype=np.float64)

    # forgot to make zero the accumulator array
    for col_ind in range(N_CIR_PERMUTATIONS):
        for row_ind in range(N_ALL_PERMUTATIONS):
            kAs[row_ind][col_ind] = 0.0

    for i in range(DIM):
        prevState[i] = curState[i]

    while (iterIndex <= nStepsAttractor) and notInEquilibrium:
        # start computing integral 
        stepper_rk4_full(params, curState, dt)
        iterIndex += 1

        # and here we will accumulate integrals
        for col_ind in range(N_CIR_PERMUTATIONS):
            # extract a permutation that corresponds to the
            # columns of detectives array
            curColPermutation[0] = cirPermutationsFlat[col_ind * DIM + 0]
            curColPermutation[1] = cirPermutationsFlat[col_ind * DIM + 1]
            curColPermutation[2] = cirPermutationsFlat[col_ind * DIM + 2]
            curColPermutation[3] = cirPermutationsFlat[col_ind * DIM + 3]

            permuteStateInto(curState, curColPermutation, startCurState)
            permuteStateInto(prevState, curColPermutation, startPrevState)

            for row_ind in range(N_ALL_PERMUTATIONS):
                curRowPermutation[0] = fullPermutationsFlat[row_ind*DIM + 0]
                curRowPermutation[1] = fullPermutationsFlat[row_ind*DIM + 1]
                curRowPermutation[2] = fullPermutationsFlat[row_ind*DIM + 2]
                curRowPermutation[3] = fullPermutationsFlat[row_ind*DIM + 3]

                permuteStateInto(startPrevState, curRowPermutation, permutedPrevState)
                permuteStateInto(startCurState, curRowPermutation, permutedCurState)

                # here we just assigned value of integral over small segment, not
                # added to an already computed integral
                kAs[row_ind][col_ind] += trapezoidRule(permutedPrevState, permutedCurState)

        # updating prevState
        for i in range(DIM):
            prevState[i] = curState[i]

        # this is a check for getting stuck into equilibrium
        full_rhs(params, curState, curRhs)
        if abs(curRhs[0]) < 1e-8 and abs(curRhs[1]) < 1e-8 and abs(curRhs[2]) < 1e-8 and abs(curRhs[3]) < 1e-8:
            notInEquilibrium = False 

    if notInEquilibrium:
        out_array[0] = SUCCESSFULLY_COMPUTED
        # don't forget to multiply all entries by dt/2 
        # and then divide by tAttr    
        # dt/2 / tAttr = dt/(2 * tAttr) = dt/(2 * dt * nStepsAttractor)= 1/(2*nStepsAttractor)
        # also сompare columns of kAs with the first one in order to get 
        # symmetry type
        for row_ind in range(N_ALL_PERMUTATIONS):
            for col_ind in range(N_CIR_PERMUTATIONS):
                kAs[row_ind][col_ind] /= (2.0 * nStepsAttractor)

        for col_ind in range(1, N_CIR_PERMUTATIONS):
            maxAbsDiff = abs(kAs[0][0] - kAs[0][col_ind])
            for row_ind in range(1, N_ALL_PERMUTATIONS):
                curAbsDiff = abs(kAs[row_ind][0] - kAs[row_ind][col_ind])
                if curAbsDiff > maxAbsDiff:
                    maxAbsDiff = curAbsDiff

            out_array[col_ind] = maxAbsDiff
    else:
        out_array[0] = STUCK_INTO_EQUILIBRIUM
        out_array[1] = -10.
        out_array[2] = -10.
        out_array[3] = -10.

@cuda.jit
def sweep_detective_threads(detectives_array,
                            inits,
                            nones,
                            params_x, 
                            params_y, 
                            def_params, 
                            param_x_ind, 
                            param_y_ind, 
                            up_n, 
                            down_n, 
                            left_n,
                            right_n, 
                            dt, 
                            nStepsSkip, 
                            nStepsAttractor,
                            fullPermutationsFlat,
                            cirPermutationsFlat):
    """CUDA kernel"""
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    grid_count = (left_n + right_n + 1) * (up_n + down_n + 1)

    if idx < grid_count:

        is_in_nones = False
        for i in range(len(nones)):
            if idx == nones[i]:
                is_in_nones = True

                # fill out first field with error code 
                # and other fields with negative garbage
                detectives_array[idx*DETECTIVE_ENTRY_SIZE] = NO_INIT_FOUND_ERROR
                detectives_array[idx*DETECTIVE_ENTRY_SIZE + 1] = -1.
                detectives_array[idx*DETECTIVE_ENTRY_SIZE + 2] = -1.
                detectives_array[idx*DETECTIVE_ENTRY_SIZE + 3] = -1.
                break
        if not is_in_nones:
            initFullSys = cuda.local.array(DIM, dtype=np.float64)
            params = cuda.local.array(PAR_SPACE_DIM, dtype=np.float64)

            # convert 3d inits of reduced system into
            # 4d init of a full system
            initFullSys[0] = 0.
            for i in range(DIM_REDUCED):
                initFullSys[i+1] = inits[idx * DIM_REDUCED + i]

            for i in range(PAR_SPACE_DIM):
                params[i] = def_params[i]
            params[param_x_ind] = params_x[idx]
            params[param_y_ind] = params_y[idx]

            out_array = cuda.local.array(DETECTIVE_ENTRY_SIZE, dtype=np.float64)
            symmetry_type_compute(initFullSys, params, dt, nStepsSkip, nStepsAttractor, out_array, fullPermutationsFlat, cirPermutationsFlat)

            #copying return of detectives_compute to an outer array 
            # print("out_array[", 0, "] = ", out_array[0])
            # print("out_array[", 1, "] = ", out_array[1])
            # print("out_array[", 2, "] = ", out_array[2])
            # print("out_array[", 3, "] = ", out_array[3])
            for i in range(DETECTIVE_ENTRY_SIZE):
                # print("out_array[", i, "] = ", out_array[i])
                detectives_array[DETECTIVE_ENTRY_SIZE*idx + i] = out_array[i]

def sweep_detective(
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
        nStepsSkip, 
        nStepsAttractor):
    total_parameter_space_size = (left_n + right_n + 1) * (up_n + down_n + 1)
    detectives_array = np.zeros(total_parameter_space_size*DETECTIVE_ENTRY_SIZE)
    detectives_array_gpu = cuda.device_array(total_parameter_space_size*DETECTIVE_ENTRY_SIZE)

    inits_gpu = cuda.to_device(inits)
    nones_gpu = cuda.to_device(nones)
    def_params_gpu = cuda.to_device(def_params)
    params_x_gpu = cuda.to_device(params_x)
    params_y_gpu = cuda.to_device(params_y)

    param_x_idx = param_to_index[param_x_str]
    param_y_idx = param_to_index[param_y_str]

    grid_x_dimension = (total_parameter_space_size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    dim_grid = grid_x_dimension
    dim_block = THREADS_PER_BLOCK

    # here we enlist all permutations of S4 in order 
    # to compute detective for an attractor
    indices = [0, 1, 2, 3]
    permArray = list(itertools.permutations(indices))
    flatPermArray = list(itertools.chain.from_iterable(permArray))
    flatPermutationsGpu = cuda.to_device(flatPermArray)

    # list specific permutations that are 
    # relevant to CIR symmetries
    cirPermutations = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]
    cirPermutationsFlat = list(itertools.chain.from_iterable(cirPermutations))
    cirPermutationsFlatGpu = cuda.to_device(cirPermutationsFlat)

    print(f"Num of blocks per grid:       {dim_grid}")
    print(f"Num of threads per block:     {dim_block}")
    print(f"Total Num of threads running: {dim_grid * dim_block}")
    print(f"Parameters a_count = {left_n + right_n + 1}, b_count = {up_n + down_n + 1}")

    # call CUDA kernel
    sweep_detective_threads[dim_grid, dim_block](  # blocks, threads
        detectives_array_gpu,
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
        nStepsSkip, 
        nStepsAttractor,
        flatPermutationsGpu,
        cirPermutationsFlatGpu
    )

    detectives_array_gpu.copy_to_host(detectives_array)

    return detectives_array


def makeRuleOne(logD01_threshold, logD02_threshold):
    def ruleOne(dists):
        logD01, logD02, logD03 = np.log10(np.array(dists))
        if logD01 < logD01_threshold or logD03 < logD01_threshold:
            return CODE_T1_SYMMETRY
        elif logD02 < logD02_threshold:
            return CODE_T2_SYMMETRY
        else:
            return CODE_T0_SYMMETRY
        
    return ruleOne




