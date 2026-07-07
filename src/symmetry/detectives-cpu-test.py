import numpy as np
# from src.cuda_sweep.sweep_fbpo import full_rhs
import itertools
from matplotlib import colors
import matplotlib.pyplot as plt

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

def full_rhs(params, phis, dphis):
    """Calculates the right-hand side of the full system"""
    w, a, b, r = params
    for i in range(4):
        dphis[i] = w
        for j in range(4):
            dphis[i] += 0.25 * (-np.sin(phis[i] - phis[j] + a) + r * np.sin(2 * (phis[i] - phis[j]) + b))

def stepper_rk4_full(params, y_curr, dt):
    """Makes RK-4 step and saves the value in y_curr"""
    k1 = np.empty(DIM, dtype=np.float64)
    k2 = np.empty(DIM, dtype=np.float64)
    k3 = np.empty(DIM, dtype=np.float64)
    k4 = np.empty(DIM, dtype=np.float64)
    y_temp = np.empty(DIM, dtype=np.float64)

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

def ashwinDetective(state):
    x1 = state[0]
    x2 = state[1]
    x3 = state[2]
    return np.sin(3*x1)*np.sin(6*x2)*np.sin(9*x3)

def trapezoidRule(prevState, curState):
    left = ashwinDetective(prevState)
    right = ashwinDetective(curState)
    return left + right

def permuteStateInto(origState, permut, permutedState):
    for ind in range(DIM):
        permutedState[ind] = origState[permut[ind]]


def symmetry_type_compute(initFullSys, params, dt, nStepsSkip, nStepsAttractor, out_array, fullPermutationsFlat, cirPermutationsFlat):
    # # TODO: delete that shit, it's just dumb useless code
    # out_array[0] = SUCCESSFULY_COMPUTED
    # out_array[1] = -10.
    # out_array[2] = -10.
    # out_array[3] = -10.

    curState = np.empty(DIM, dtype=np.float64)
    prevState = np.empty(DIM, dtype=np.float64)
    startCurState = np.empty(DIM, dtype=np.float64)
    startPrevState = np.empty(DIM, dtype=np.float64)
    permutedCurState = np.empty(DIM, dtype=np.float64)
    permutedPrevState = np.empty(DIM, dtype=np.float64)
    curRhs = np.empty(DIM, dtype=np.float64)

    curRowPermutation = np.empty(DIM, dtype=np.int16)
    curColPermutation = np.empty(DIM, dtype=np.int16)

    notInEquilibrium = True

    for i in range(DIM):
        curState[i] = initFullSys[i]

    # first we compute the transient before we get onto an attractor
    iterIndex = 0
    while (iterIndex <= nStepsSkip) and notInEquilibrium:
        stepper_rk4_full(params, curState, dt)
        iterIndex += 1

        # this is a check for getting stuck into equilibrium
        full_rhs(params, curState, curRhs)
        if abs(curRhs[0]) < 1e-8 and abs(curRhs[1]) < 1e-8 and abs(curRhs[2]) < 1e-8 and abs(curRhs[3]) < 1e-8:
            notInEquilibrium = False


    # and now we compute on the attractor
    iterIndex = 0
    kAs = np.zeros((N_ALL_PERMUTATIONS, N_CIR_PERMUTATIONS), dtype=np.float64)

    for i in range(DIM):
        prevState[i] = curState[i]

    while (iterIndex <= nStepsAttractor) and notInEquilibrium:
        print(f"Computing detectives: {iterIndex = }")
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

                # print(f"{ashwinDetective(permutedPrevState) = }")
                # print(f"{ashwinDetective(permutedCurState) = }")
                segmentIntegralValue = trapezoidRule(permutedPrevState, permutedCurState)
                assert not np.isnan(segmentIntegralValue)
                kAs[row_ind][col_ind] += segmentIntegralValue
                assert not np.isnan(kAs[row_ind][col_ind])
                # print(f"{kAs[row_ind][col_ind] = }")

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


if __name__ == '__main__':
    # # Fig. 8d chaos T4
    # a = -2.484
    # b = -1.61268422884276
    # r = 1.0
    # x = 1.422975473348162
    # y = 3.174906328214909
    # z = 4.543547968780502

    # Fig. 8k chaos T2
    a = -2.837
    b = -1.61268422884276
    r = 1.0
    x = 1.688590742253721
    y = 3.234142613840711
    z = 4.803704960201133

    # # Fig. 8l chaos T1
    # a = -2.84
    # b = -1.61268422884276
    # r = 1.0
    # x = 1.535559500622468
    # y = 3.114435506361571
    # z = 4.551518563147478

    initFullSys = [0, x, y, z]
    params = [0, a, b, r]
    out_array = np.empty(DETECTIVE_ENTRY_SIZE, dtype=np.float64)
    dt = 0.01
    nStepsSkip = 0
    nStepsAttractor = 10000_00

    indices = [0, 1, 2, 3]
    permArray = list(itertools.permutations(indices))
    flatPermArray = list(itertools.chain.from_iterable(permArray))

    cirPermutations = [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]
    cirPermutationsFlat = list(itertools.chain.from_iterable(cirPermutations))
    
    testArr = np.empty(4, dtype=np.float64)
    print(f"{testArr = }")

    symmetry_type_compute(initFullSys,params, dt, nStepsSkip, nStepsAttractor,out_array, flatPermArray, cirPermutationsFlat)

    print(f"{list(out_array) = }")
