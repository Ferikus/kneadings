import numpy as np
import eq_finder.systems_fun as sf
import eq_finder.SystOsscills as so
import scipy

from system_analysis.system import DIM_REDUCED  # ???

# 1. найти начальные точки для всей сетки параметров
#    1.1 реализовать поиск седло-фокусов
#        c проверкой на нахождение внутри тетраэдра
#    1.2 протянуть его по параметрам
#    1.3 для всех существующих узлов найти начальные точки
#    1.4 организовать передачу массива нач точек в код с нидингами
# 2. нарисовать временные реализации рядом с особым фокусом
# 3. подстроить функцию интегратора под свою систему
#    (с зависимостью от координат с.р.?)

w = 0
r = 1.0


def find_equilibrium_by_guess(sys, initial_guess=np.zeros(3), tol=1e-10):  # max_iter=100
    """Находит состояние равновесия системы для заданных параметров."""
    def func(psi):
        return np.asarray(sys.getReducedSystem(psi))

    initial_guess = np.asarray(initial_guess)

    result = scipy.optimize.root(
        func,
        initial_guess,
        method='krylov',
        options={'xtol': tol}
    )
    if not result.success:
        return None

    eq_coords = result.x
    eq_obj = sf.getEquilibriumInfo(eq_coords, sys.getReducedSystemJac)
    return eq_obj


def continue_equilibrium(start_a, start_b, start_eq_coords,
                           up_n, down_n, left_n, right_n,
                           up_step, down_step, left_step, right_step):
    rows = up_n + down_n + 1
    cols = left_n + right_n + 1
    # в каждой точке сетки есть словарь с параметрами, с.р. и н.у.
    # grid = [[{'params': None, 'eq_obj': None} for _ in range(cols)] for _ in range(rows)]
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    print(len(grid), len(grid[0]))

    # начало координат слева снизу
    # обработка стартовой точки
    start_row, start_col = down_n, left_n
    start_sys = so.FourBiharmonicPhaseOscillators(w, start_a, start_b, r)
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

    # находим с.р. и н.у. для них в каждой точке
    for di, dj in deltas:
        curr_i = start_col + di
        curr_j = start_row + dj

        da = di * (up_step if di > 0 else down_step)
        db = dj * (right_step if dj > 0 else left_step)

        curr_a = start_a + da
        curr_b = start_b + db

        sys = so.FourBiharmonicPhaseOscillators(w, curr_a, curr_b, r)

        neighbors = []
        for ni, nj in [(curr_i - 1, curr_j), (curr_i + 1, curr_j),
                       (curr_i, curr_j - 1), (curr_i, curr_j + 1)]:
            if 0 <= ni < cols and 0 <= nj < rows and grid[nj][ni] is not None:
                neighbors.append(grid[nj][ni].coordinates)

        eq_obj = None
        for guess in neighbors:
            eq_obj = find_equilibrium_by_guess(sys, initial_guess=guess)
            if eq_obj is not None:
                break

        if eq_obj is not None:
            grid[curr_j][curr_i] = eq_obj
            print(
                f"Node ({curr_i}, {curr_j}) | Equilibrium {eq_obj.coordinates} was found "
                f"with parameters ({curr_a:.3f}, {curr_b:.3f})")
        else:
            print(
                f"Node ({curr_i}, {curr_j}) | No equilibrium was found "
                f"with parameters ({curr_a:.3f}, {curr_b:.3f})")

    return grid


def get_saddle_foci_grid(grid):
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
    # допустим, что grid на самом деле это просто двумерный массив с eq_objs
    # тогда мы можем записывать inits сразу в одномерный массив
    # опираясь только на i j в двумерном массиве grid
    print("\nFinding initial conditions...")

    def separation_condition(pt, eq_pt):
        """При поиске н.у. выбираем те точки на сепаратрисах, которые лежат в СIR"""
        return sf.isInCIR(pt, strictly=True)

    inits = np.empty(DIM_REDUCED * (left_n + right_n + 1) * (up_n + down_n + 1))
    nones = []  # массив индексов там, где None. Нужен при обходе нидингов

    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            index = i + j * (left_n + right_n + 1)

            eq_obj = sf_grid[j][i]
            if eq_obj is not None:
                if sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, separation_condition, sf.STD_PRECISION):
                    # print(sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, separation_condition, sf.STD_PRECISION))
                    init_pt = sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, separation_condition, sf.STD_PRECISION)[0]
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


def generate_parameters(start_a, start_b, up_n, down_n, left_n, right_n,
                        up_step, down_step, left_step, right_step):
    print("\nGenerating parameters...")
    alphas = np.empty((left_n + right_n + 1) * (up_n + down_n + 1))
    betas = np.empty((left_n + right_n + 1) * (up_n + down_n + 1))

    for j in range(up_n + down_n + 1):
        for i in range(left_n + right_n + 1):
            index = i + j * (left_n + right_n + 1)

            da = (i - left_n) * (right_step if i > left_n else left_step)
            db = (j - down_n) * (up_step if j > down_n else down_step)

            alphas[index] = start_a + da
            betas[index] = start_b + db
            print(f"alpha_{index} {i, j} {alphas[index]}")
            print(f"beta_{index} {i, j} {betas[index]}")

    return alphas, betas


# def explore_parameter_grid(start_a, start_b, start_eq_coords,
#                            up_n, down_n, left_n, right_n,
#                            up_step, down_step, left_step, right_step):
#     def separation_condition(pt, eq_pt):
#         """При поиске н.у. выбираем те точки на сепаратрисах, которые лежат в СIR"""
#         return sf.isInCIR(pt, strictly=True)
#
#     rows = up_n + down_n + 1
#     cols = left_n + right_n + 1
#     # в каждой точке сетки есть словарь с параметрами, с.р. и н.у.
#     grid = [[{'params': None, 'equilibrium': None, 'init_pt': None} for _ in range(cols)] for _ in range(rows)]
#
#     # начало координат слева снизу
#     # обработка стартовой точки
#     start_row, start_col = down_n, left_n
#     grid[start_row][start_col]['params'] = (start_a, start_b)
#     grid[start_row][start_col]['equilibrium'] = start_eq_coords
#
#     start_sys = so.FourBiharmonicPhaseOscillators(w, start_a, start_b, r)
#     start_eq_obj = sf.getEquilibriumInfo(start_eq_coords, start_sys.getReducedSystemJac)
#
#     if sf.getInitPointsOnUnstable1DSeparatrix(start_eq_obj, separation_condition, sf.STD_PRECISION):
#         # print(sf.getInitPointsOnUnstable1DSeparatrix(start_eq_obj, separation_condition, sf.STD_PRECISION))
#         start_init_pt = sf.getInitPointsOnUnstable1DSeparatrix(start_eq_obj, separation_condition, sf.STD_PRECISION)[0]
#     else:
#         start_init_pt = None
#     grid[start_row][start_col]['init_pt'] = start_init_pt
#
#     # список всех возможных смещений
#     deltas = []
#     for di in range(-down_n, up_n + 1):
#         for dj in range(-left_n, right_n + 1):
#             if (di, dj) == (0, 0):
#                 continue
#             deltas.append((di, dj))
#
#     # сортируем по удалению от центра (сначала ближайшие точки)
#     deltas.sort(key=lambda delta: abs(delta[0]) + abs(delta[1]))
#
#     # находим с.р. и н.у. для них в каждой точке
#     for di, dj in deltas:
#         curr_i = start_row + di
#         curr_j = start_col + dj
#
#         da = di * (up_step if di > 0 else down_step)
#         db = dj * (right_step if dj > 0 else left_step)
#
#         curr_a = start_a + da
#         curr_b = start_b + db
#         grid[curr_i][curr_j]['params'] = (curr_a, curr_b)
#
#         sys = so.FourBiharmonicPhaseOscillators(w, curr_a, curr_b, r)
#
#         neighbors = []
#         for ni, nj in [(curr_i - 1, curr_j), (curr_i + 1, curr_j),
#                        (curr_i, curr_j - 1), (curr_i, curr_j + 1)]:
#             if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj]['equilibrium'] is not None:
#                 neighbors.append(grid[ni][nj]['equilibrium'])
#
#         equilibrium = None
#         for guess in neighbors:
#             eq_obj = find_equilibrium_by_guess(sys, initial_guess=guess)
#             if eq_obj is not None:
#                 equilibrium = np.asarray(eq_obj.coordinates)
#                 break
#
#         if eq_obj is not None:
#             grid[curr_i][curr_j]['equilibrium'] = equilibrium
#
#             if sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, separation_condition, sf.STD_PRECISION):
#                 # print(sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, separation_condition, sf.STD_PRECISION))
#                 init_pt = sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, separation_condition, sf.STD_PRECISION)[0]
#             else:
#                 init_pt = None
#             grid[curr_i][curr_j]['init_pt'] = init_pt
#
#             print(
#                 f"Node ({curr_i}, {curr_j}) | Saddle-focus {equilibrium.round(4)} was found "
#                 f"with parameters ({curr_a:.3f}, {curr_b:.3f})")
#         else:
#             print(
#                 f"Node ({curr_i}, {curr_j}) | No saddle-focus was found "
#                 f"with parameters ({curr_a:.3f}, {curr_b:.3f})")
#
#     return grid


if __name__ == '__main__':
    # начальные параметры (w и r объявлены глобально)
    a = -2.67
    b = -1.61268422884276

    start_sys = so.FourBiharmonicPhaseOscillators(w, a, b, r)

    def rhs_wrapper(psis):
        return start_sys.getReducedSystem(psis)

    def rhs_jac_wrapper(psis):
        return start_sys.getReducedSystemJac(psis)

    # поиск стартового седло-фокуса
    bounds = [(-0.1, 2 * np.pi + 0.1)] * 3
    borders = [(-1e-15, 2 * np.pi + 1e-15)] * 3

    equilibria = sf.findEquilibria(rhs_wrapper, rhs_jac_wrapper, rhs_wrapper, rhs_jac_wrapper,
                                   lambda phi: phi, bounds, borders,
                                   sf.ShgoEqFinder(1000, 1, 1e-10),
                                   sf.STD_PRECISION)

    start_eq = None
    for eq in equilibria:  # перебираем все с.р., которые были найдены
        print(f"{sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION)} at {eq.coordinates}")
        if sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION) and sf.isInCIR(eq.coordinates, strictly=False):
            # ЗАМЕНИТЬ НА УСЛОВИЕ НА ГРАНИЦЕ ВМЕСТО ВСЕЙ ОБЛАСТИ
            start_eq = np.array(eq.coordinates)
            print(f"\nStarting with saddle-focus {start_eq.round(4)} with parameters ({w:.3f}, {a:.3f}, {b:.3f}, {r:.3f})")
            break

    up_n = 1
    down_n = 2
    left_n = 1
    right_n = 1

    up_step = 0.01
    down_step = 0.01
    left_step = 0.01
    right_step = 0.01

    if start_eq is not None:
        eq_grid = continue_equilibrium(a, b, start_eq, up_n, down_n, left_n, right_n,
                                       up_step, down_step, left_step, right_step)
        sf_grid = get_saddle_foci_grid(eq_grid)
        inits, nones = find_inits_for_equilibrium_grid(sf_grid)
        alphas, betas = generate_parameters(a, b, up_n, down_n, left_n, right_n,
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

    # if start_eq is not None:
    #     result_grid = explore_parameter_grid(a, b, start_eq,
    #                                          up_n, down_n, left_n, right_n,
    #                                          up_step, down_step, left_step, right_step)
    # else:
    #     print("Start saddle-focus was not found")

    # inits = np.empty(DIM_REDUCED * (left_n + right_n + 1) * (up_n + down_n + 1))
    # nones = []
    # alphas = np.empty((left_n + right_n + 1) * (up_n + down_n + 1))
    # betas = np.empty((left_n + right_n + 1) * (up_n + down_n + 1))
    # for i in range(left_n + right_n + 1):
    #     for j in range(up_n + down_n + 1):
    #         index = j + i * (left_n + right_n + 1)
    #         alphas[index] = result_grid[j][i]['params'][0]
    #         betas[index] = result_grid[j][i]['params'][1]
    #         if result_grid[j][i]['init_pt'] is not None:
    #             inits[index * DIM_REDUCED] = result_grid[j][i]['init_pt'][0]
    #             inits[index * DIM_REDUCED + 1] = result_grid[j][i]['init_pt'][1]
    #             inits[index * DIM_REDUCED + 2] = result_grid[j][i]['init_pt'][2]
    #         else:
    #             nones.append(index)  # массив индексов там, где None
    #
    # print(f'Initial conditions: {inits}')
    # print(f'Alphas: {alphas}')
    # print(f'Betas: {betas}')
    # print(f'Nones: {nones}')