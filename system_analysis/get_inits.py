import numpy as np
import eq_finder.systems_fun as sf
import eq_finder.SystOsscills as so
import scipy

from system_analysis.system import DIM_REDUCED  # ???


def find_equilibrium_by_guess(rhs, jac, initial_guess=np.zeros(3), tol=1e-10): # sys
    """Находит состояние равновесия системы для заданных параметров."""
    initial_guess = np.asarray(initial_guess)

    result = scipy.optimize.root(
        rhs,
        initial_guess,
        method='krylov',
        options={'xtol': tol}
    )
    if not result.success:
        return None

    eq_coords = result.x
    eq_obj = sf.getEquilibriumInfo(eq_coords, jac)
    return eq_obj


def continue_equilibrium(create_sys, rhs, jac, start_grid_params, other_params, start_eq_coords,
                         up_n, down_n, left_n, right_n,
                         up_step, down_step, left_step, right_step):
    """Продолжает состояние равновесия по сетке параметров"""
    rows = up_n + down_n + 1
    cols = left_n + right_n + 1
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    print(len(grid), len(grid[0]))

    # начало координат слева снизу
    # обработка стартовой точки
    start_row, start_col = down_n, left_n
    start_sys = create_sys(start_grid_params, other_params)
    start_eq_obj = sf.getEquilibriumInfo(start_eq_coords, start_sys.getReducedSystemJac)
    grid[down_n][left_n] = start_eq_obj

    # список всех возможных смещений
    deltas = []
    for dj in range(-down_n, up_n + 1):
        for di in range(-left_n, right_n + 1):
            if (di, dj) == (0, 0):
                continue
            deltas.append((di, dj))

    # сортируем по удалению от центра (сначала ближайшие точки)
    deltas.sort(key=lambda delta: abs(delta[0]) + abs(delta[1]))

    # находим с.р. в каждой точке
    for di, dj in deltas:
        curr_i = start_col + di
        curr_j = start_row + dj

        da = di * (up_step if di > 0 else down_step)
        db = dj * (right_step if dj > 0 else left_step)

        curr_grid_params = (start_grid_params[0] + da, start_grid_params[1] + db)

        sys = create_sys(curr_grid_params, other_params)

        neighbors = []
        for ni, nj in [(curr_i - 1, curr_j), (curr_i + 1, curr_j),
                       (curr_i, curr_j - 1), (curr_i, curr_j + 1)]:
            if 0 <= ni < cols and 0 <= nj < rows and grid[nj][ni] is not None:
                neighbors.append(grid[nj][ni].coordinates)

        eq_obj = None
        for guess in neighbors:
            eq_obj = find_equilibrium_by_guess(lambda args: rhs(sys, args), lambda args: jac(sys, args), initial_guess=guess)
            if eq_obj is not None:
                break

        if eq_obj is not None:
            grid[curr_j][curr_i] = eq_obj
            print(
                f"Node ({curr_i}, {curr_j}) | Equilibrium {eq_obj.coordinates} was found "
                f"with parameters ({curr_grid_params[0]:.3f}, {curr_grid_params[1]:.3f})")
        else:
            print(
                f"Node ({curr_i}, {curr_j}) | No equilibrium was found "
                f"with parameters ({curr_grid_params[0]:.3f}, {curr_grid_params[1]:.3f})")

    return grid


def get_saddle_foci_grid(grid):
    """Составляет сетку седло-фокусов по сетке состояний равновесия"""
    print("\nFilling up saddle-foci grid...")
    sf_grid = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]
    print(len(sf_grid), len(sf_grid[0]))

    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            if grid[j][i] is not None and sf.is3DSaddleFocusWith1dU(grid[j][i], sf.STD_PRECISION):
                sf_grid[j][i] = grid[j][i]
                print(f"{i + j * (left_n + right_n + 1)} {i, j} -- saddle-focus")
            else:
                print(f"{i + j * (left_n + right_n + 1)} {i, j} -- none")

    return sf_grid


def find_inits_for_equilibrium_grid(sf_grid):
    """Находит начальные условия для сетки седло-фокусов"""
    # допустим, что grid на самом деле это просто двумерный массив с eq_objs
    # тогда мы можем записывать inits сразу в одномерный массив
    # опираясь только на i j в двумерном массиве grid
    print("\nFinding initial conditions...")

    inits = np.empty(DIM_REDUCED * (left_n + right_n + 1) * (up_n + down_n + 1))
    nones = []  # массив индексов там, где None. Нужен при обходе нидингов

    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            index = i + j * (left_n + right_n + 1)

            eq_obj = sf_grid[j][i]
            if eq_obj is not None:
                if sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, sf.pickCirSeparatrix, sf.STD_PRECISION):
                    init_pt = sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, sf.pickCirSeparatrix, sf.STD_PRECISION)[0]
                    inits[index * DIM_REDUCED] = init_pt[0]
                    inits[index * DIM_REDUCED + 1] = init_pt[1]
                    inits[index * DIM_REDUCED + 2] = init_pt[2]
                    print(f"{index} {init_pt}")
                else:
                    nones.append(index)
                    print(f"{index} {i, j} None: no initial condition was found for the saddle-focus")
            else:
                nones.append(index)
                print(f"{index} {i, j} None: no equilibrium object")

    print(f"\nNones: {nones}")

    return inits, nones


def generate_parameters(start_params, up_n, down_n, left_n, right_n,
                        up_step, down_step, left_step, right_step):
    """Генерирует массивы параметров для последующего подсчёта нидингов"""
    print("\nGenerating parameters...")
    start_params_x = np.empty((left_n + right_n + 1) * (up_n + down_n + 1))
    start_params_y = np.empty((left_n + right_n + 1) * (up_n + down_n + 1))

    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            index = i + j * (left_n + right_n + 1)

            da = (i - left_n) * (right_step if i > left_n else left_step)
            db = (j - down_n) * (up_step if j > down_n else down_step)

            start_params_x[index] = start_params[0] + da
            start_params_y[index] = start_params[1] + db
            print(f"param1_{index} {i, j} {start_params_x[index]}")
            print(f"param2_{index} {i, j} {start_params_y[index]}")

    return start_params_x, start_params_y


if __name__ == '__main__':
    a = -2.67
    b = -1.61268422884276
    w = 0
    r = 1.0

    start_sys = so.FourBiharmonicPhaseOscillators(w, a, b, r)

    # поиск стартового седло-фокуса
    bounds = [(-0.1, 2 * np.pi + 0.1)] * 2
    borders = [(-1e-15, 2 * np.pi + 1e-15)] * 2

    # первые две функции -- общая система, вторые две -- в которой ищем с.р., дальше функция приведения
    equilibria = sf.findEquilibria(lambda psis: start_sys.getReducedSystem(psis), lambda psis: start_sys.getReducedSystemJac(psis),
                                   lambda psis: start_sys.getRestriction(psis), lambda psis: start_sys.getRestrictionJac(psis),
                                   lambda phi: np.concatenate([[0.], phi]), bounds, borders,
                                   sf.ShgoEqFinder(1000, 1, 1e-10),
                                   sf.STD_PRECISION)

    start_eq = None
    for eq in equilibria:  # перебираем все с.р., которые были найдены
        print(f"{sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION)} at {eq.coordinates}")
        if sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION):
            start_eq = np.array(eq.coordinates)
            print(f"\nStarting with saddle-focus {start_eq.round(4)} with parameters ({w:.3f}, {a:.3f}, {b:.3f}, {r:.3f})")
            break

    up_n = 20
    down_n = 20
    left_n = 20
    right_n = 20

    up_step = 0.01
    down_step = 0.01
    left_step = 0.01
    right_step = 0.01

    def create_fbpo_system(grid_params, other_params):
        a, b = grid_params
        w, r = other_params
        return so.FourBiharmonicPhaseOscillators(w, a, b, r)

    def reduced_rhs_wrapper(sys, psis):
        return sys.getReducedSystem(psis)

    def reduced_jac_wrapper(sys, psis):
        return sys.getReducedSystemJac(psis)

    if start_eq is not None:
        eq_grid = continue_equilibrium(create_fbpo_system, reduced_rhs_wrapper, reduced_jac_wrapper,
                                       (a, b), (w, r),
                                       start_eq, up_n, down_n, left_n, right_n,
                                       up_step, down_step, left_step, right_step)
        sf_grid = get_saddle_foci_grid(eq_grid)
        inits, nones = find_inits_for_equilibrium_grid(sf_grid)
        alphas, betas = generate_parameters((a, b), up_n, down_n, left_n, right_n,
                                            up_step, down_step, left_step, right_step)
    else:
        print("Start saddle-focus was not found")

    np.savez(
        'inits.npz',
        inits=inits,
        nones=nones,
        alphas=alphas,
        betas=betas,
        up_n=up_n,
        down_n=down_n,
        left_n=left_n,
        right_n=right_n
    )