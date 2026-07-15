import numpy as np
# import h5py  # OSError: exception: access violation reading 0x0000000000000009
from os.path import join
from numba import cuda
from datetime import datetime

import lib.eq_finder.systems_fun as sf
import lib.eq_finder.SystOsscills as so
from src.system_analysis.find_equilibrium import correct_equilibrium_coords
from src.cuda_sweep.sweep_fbpo import stepper_rk4, reduced_rhs

DIM_REDUCED = 3
PARAMS_COUNT = 4
THREADS_PER_BLOCK = 512
INFINITY = 20.0
IN_EQUILIBRIUM = 1e-2


def get_manifold_vectors(eq_obj, mode):
    """Returns basis of a 2d manifold"""
    eigenvalues = np.array(eq_obj.eigenvalues)
    vectors = np.array(eq_obj.eigvectors)

    if mode == 'unstable':
        indices = np.where(eigenvalues.real > 1e-8)[0]
    else:
        indices = np.where(eigenvalues.real < -1e-8)[0]

    if len(indices) < 2:
        raise ValueError(f"Not enough eigenvectors for 2d {mode} manifold")

    if np.abs(eigenvalues[indices[0]].imag) > 1e-10:
        v1 = vectors[indices[0]].real
        v2 = vectors[indices[0]].imag
    else:
        v1 = vectors[indices[0]].real
        v2 = vectors[indices[1]].real

    v1 /= np.linalg.norm(v1)
    v2 = v2 - np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2)

    return v1, v2


@cuda.jit
def sweep_threads_check_heteroclinic(inits, params, dt, n, stride, final_pt_info):
    """CUDA kernel for checking heteroclinic connection"""
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # cuda.grid(1)
    if idx < inits.shape[0]:
        y_curr = cuda.local.array(DIM_REDUCED, dtype=np.float64)
        rhs = cuda.local.array(DIM_REDUCED, dtype=np.float64)

        for i in range(DIM_REDUCED):
            y_curr[i] = inits[idx, i]

        is_eq = 0.0
        for i in range(n):
            for j in range(stride):
                stepper_rk4(params, y_curr, dt)

            out_of_bounds = False
            for j in range(DIM_REDUCED):
                if abs(y_curr[j]) > INFINITY:
                    break
            if out_of_bounds:
                break

            reduced_rhs(params, y_curr, rhs)
            if abs(rhs[0]) < IN_EQUILIBRIUM and abs(rhs[1]) < IN_EQUILIBRIUM and abs(rhs[2]) < IN_EQUILIBRIUM:
                is_eq = 1.0
                break

        final_pt_info[idx, 0] = y_curr[0]
        final_pt_info[idx, 1] = y_curr[1]
        final_pt_info[idx, 2] = y_curr[2]
        final_pt_info[idx, 3] = is_eq


def sweep_check_heteroclinic(inits, params, dt, n, stride):
    num_points = inits.shape[0]
    inits_gpu = cuda.to_device(inits)
    params_gpu = cuda.to_device(params)
    results_gpu = cuda.device_array((num_points, 4), dtype=np.float64)

    blocks = (num_points + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    sweep_threads_check_heteroclinic[blocks, THREADS_PER_BLOCK](inits_gpu, params_gpu, dt, n, stride, results_gpu)

    return results_gpu.copy_to_host()


if __name__ == "__main__":
    params = np.array([0.0, -2.8785920, -1.6788497, 1.0])
    sys = so.FourBiharmonicPhaseOscillators(*params)
    rhs_wrapper = sys.getReducedSystem
    jac_wrapper = sys.getReducedSystemJac

    raw_sf = [1.5296057594086407, 3.2309560665295027, 4.3236725233743]
    sf_coords = correct_equilibrium_coords(rhs_wrapper, jac_wrapper, raw_sf, tol=1e-13)
    eq_sf = sf.getEquilibriumInfo(sf_coords, jac_wrapper)

    eps = 0.1  # размер окрестности
    p = 10000  # число разбиений на окружности
    dt = 0.01  # шаг интегрирования
    n = 1000000  # количество шагов интегрирования
    stride = 1  # количество шагов интегрировния за раз

    # eps = 0.05  # размер окрестности
    # p = 50000  # число разбиений на окружности
    # dt = 0.01  # шаг интегрирования
    # n = 1000000  # количество шагов интегрирования
    # stride = 1  # количество шагов интегрировния за раз

    u1, u2 = get_manifold_vectors(eq_sf, mode='unstable')  # неустойчивое многообразие седло-фокуса

    angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
    inits = np.array([sf_coords + eps * (np.cos(a) * u1 + np.sin(a) * u2) for a in angles])

    # print(f"Start saddle-Focus:\n"
    #       f"{sf_coords}\n")
    #
    # params_string = (f"Parameters:\n"
    #                  f"eps = {eps}\np = {p}\ndt = {dt}\nn = {n}\nstride = {stride}\n")
    # print(params_string)

    results = sweep_check_heteroclinic(inits, params, dt, n, stride)

    records = f"Start saddle-Focus:\n" \
              f"{sf_coords}\n" \
              f"Parameters:\n" \
              f"eps = {eps}\np = {p}\ndt = {dt}\nn = {n}\nstride = {stride}\n" \
              f"{'Index':<{len(str(p))}} | {'Angle':<21} | {'Init Pt':<64} | {'Equilibrium':<32} | {'Type':<15} | {'W^s Residual'}"
    print(records)
    for i in range(len(inits)):
        init_pt = inits[i]
        final_pt = results[i, :3]
        is_eq = bool(results[i, 3])

        if is_eq:
            target_eq_coords = correct_equilibrium_coords(rhs_wrapper, jac_wrapper, final_pt, tol=1e-13)
            target_eq = sf.getEquilibriumInfo(target_eq_coords, jac_wrapper)
            target_eq_type = target_eq.getEqType(sf.STD_PRECISION)

            if sf.is3DSaddleWith1dU(target_eq, sf.STD_PRECISION):
                s1, s2 = get_manifold_vectors(target_eq, mode='stable')  # устойчивое для седла
                normal_s = np.cross(s1, s2)  # нормаль к плоскости седла (коэффициенты аппроксимации плоскостью
                residual = np.dot(final_pt - target_eq_coords, normal_s)  # значение невязки в последней точке и в седле

                record = (f"{i:<{len(str(p))}} | {angles[i]:<21} | {str(list(init_pt)):<64} | "
                          f"{str(list(np.round(target_eq_coords, 6))):<32} | {str(target_eq_type):<15} | {residual:.6e}")
                records += record + "\n"
                print(record)

    output_dir = "../../output/check_heteroclinic/"
    mask = "check_heteroclinic"
    time_stamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    txt_outname = join(output_dir, f"{mask}_{time_stamp}.txt")
    with open(txt_outname, 'w') as txt_output:
        txt_output.write(records)
    print("Text records successfully saved")

    # hdf5_outname = join(output_dir, f"{mask}_{time_stamp}.hdf5")
    # with h5py.File(hdf5_outname, 'w') as main_folder:
    #     main_folder.create_dataset('angles', data=angles)
    #     main_folder.create_dataset('inits', data=inits)
    #     main_folder.create_dataset('results', data=results)
    #     main_folder.create_dataset('records', data=records)
    #     main_folder.attrs['params_string'] = params_string
    # print("Dataset successfully saved")
