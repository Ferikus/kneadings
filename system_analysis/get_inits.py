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


def find_equilibrium_by_guess(sys, initial_guess=np.zeros(3), tol=1e-10, max_iter=100):
    """Находит состояние равновесия системы для заданных параметров."""

    def func(psi):
        return np.asarray(sys.getReducedSystem(psi))

    def fprime(psi):
        return np.asarray(sys.getReducedSystemJac(psi))

    initial_guess = np.asarray(initial_guess)

    result = scipy.optimize.root(
        func,
        initial_guess,
        jac=fprime,
        method='hybr',
        options={'xtol': tol, 'maxfev': max_iter}
    )
    if not result.success:
        return None
    equilibrium = result.x
    # equilibrium = scipy.optimize.newton(func, initial_guess, fprime=fprime, tol=tol, maxiter=max_iter)

    # if np.linalg.norm(func(equilibrium)) < tol:  # лишнее?
    eigvals, eigvecs = np.linalg.eig(sys.getReducedSystemJac(equilibrium))
    eq_obj = sf.Equilibrium(equilibrium, eigvals, eigvecs)
    if sf.is3DSaddleFocusWith1dU(eq_obj, sf.STD_PRECISION):
        return eq_obj
    return None


def explore_parameter_grid(start_a, start_b, start_eq_coords,
                           up_n, down_n, left_n, right_n,
                           up_step, down_step, left_step, right_step):
    """Обходит сетку параметров и ищет состояния равновесия типа седло-фокус."""

    def separation_condition(pt, eq_pt):
        """Выбираем обе сепаратрисы при поиске н.у."""
        return True

    rows = up_n + down_n + 1
    cols = left_n + right_n + 1
    # в каждой точке сетки есть словарь с параметрами, с.р. и н.у.
    grid = [[{'params': None, 'equilibrium': None, 'init_pt': None} for _ in range(cols)] for _ in range(rows)]

    # начало координат слева снизу
    # обработка стартовой точки
    start_row, start_col = down_n, left_n
    grid[start_row][start_col]['params'] = (start_a, start_b)
    grid[start_row][start_col]['equilibrium'] = start_eq_coords

    start_sys = so.FourBiharmonicPhaseOscillators(w, start_a, start_b, r)
    start_eigvals, start_eigvecs = np.linalg.eig(start_sys.getReducedSystemJac(start_eq_coords))
    start_eq_obj = sf.Equilibrium(start_eq_coords, start_eigvals, start_eigvecs)
    start_init_pt = sf.getInitPointsOnUnstable1DSeparatrix(start_eq_obj, separation_condition, sf.STD_PRECISION)[0]
    grid[start_row][start_col]['init_pt'] = start_init_pt

    # список всех возможных смещений
    deltas = []
    for di in range(-down_n, up_n + 1):
        for dj in range(-left_n, right_n + 1):
            if (di, dj) == (0, 0):
                continue
            deltas.append((di, dj))

    # сортируем по удалению от центра (сначала ближайшие точки)
    deltas.sort(key=lambda delta: abs(delta[0]) + abs(delta[1]))

    # находим с.р. и н.у. для них в каждой точке
    for di, dj in deltas:
        curr_i = start_row + di
        curr_j = start_col + dj

        # if 0 <= curr_i < rows and 0 <= curr_j < cols:  вроде это условие уже предусмотрено???
        da = di * (up_step if di > 0 else down_step)
        db = dj * (right_step if dj > 0 else left_step)

        curr_a = start_a + da
        curr_b = start_b + db
        grid[curr_i][curr_j]['params'] = (curr_a, curr_b)

        sys = so.FourBiharmonicPhaseOscillators(w, curr_a, curr_b, r)

        neighbors = []
        for ni, nj in [(curr_i - 1, curr_j), (curr_i + 1, curr_j),
                       (curr_i, curr_j - 1), (curr_i, curr_j + 1)]:
            if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj]['equilibrium'] is not None:
                neighbors.append(grid[ni][nj]['equilibrium'])

        equilibrium = None
        for guess in neighbors:
            eq_obj = find_equilibrium_by_guess(sys, initial_guess=guess)
            if eq_obj is not None:
                equilibrium = np.asarray(eq_obj.coordinates)
                break

        if eq_obj is not None:
            grid[curr_i][curr_j]['equilibrium'] = equilibrium

            init_pt = sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, separation_condition, sf.STD_PRECISION)[0]
            # берём pt1 = (eq.coordinates + unstVector * ps.separatrixShift).real
            grid[curr_i][curr_j]['init_pt'] = init_pt

            print(
                f"Node ({curr_i}, {curr_j}) | Saddle-focus {equilibrium.round(4)} was found "
                f"with parameters ({curr_a:.3f}, {curr_b:.3f})")
        else:
            print(
                f"Node ({curr_i}, {curr_j}) | No saddle-focus was found "
                f"with parameters ({curr_a:.3f}, {curr_b:.3f})")

    return grid


if __name__ == '__main__':
    # начальные параметры (w и r объявлены глобально)
    a = -2.911209192326542
    b = -1.612684228842761

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
        if sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION):
            start_eq = np.array(eq.coordinates)
            print(f"Starting with saddle-focus {start_eq.round(4)} with parameters ({w:.3f}, {a:.3f}, {b:.3f}, {r:.3f})")
            break

    up_n = 1
    down_n = 1
    left_n = 1
    right_n = 1

    up_step = 0.01
    down_step = 0.01
    left_step = 0.01
    right_step = 0.01

    if start_eq is not None:
        result_grid = explore_parameter_grid(a, b, start_eq,
                                             up_n, down_n, left_n, right_n,
                                             up_step, down_step, left_step, right_step)
    else:
        print("Start saddle-focus was not found")

    inits = np.empty(DIM_REDUCED * (left_n + right_n + 1) * (up_n + down_n + 1))
    nones = []
    alphas = np.empty((left_n + right_n + 1) * (up_n + down_n + 1))
    betas = np.empty((left_n + right_n + 1) * (up_n + down_n + 1))
    for i in range(left_n + right_n + 1):
        for j in range(up_n + down_n + 1):
            index = j + i * (left_n + right_n + 1)
            alphas[index] = result_grid[j][i]['params'][0]
            betas[index] = result_grid[j][i]['params'][1]
            if result_grid[j][i]['init_pt'] is not None:
                inits[index * DIM_REDUCED] = result_grid[j][i]['init_pt'][0]
                inits[index * DIM_REDUCED + 1] = result_grid[j][i]['init_pt'][1]
                inits[index * DIM_REDUCED + 2] = result_grid[j][i]['init_pt'][2]
            else:
                nones.append(index)  # массив индексов там, где None

    print(f'Initial conditions: {inits}')
    print(f'Alphas: {alphas}')
    print(f'Betas: {betas}')
    print(f'Nones: {nones}')