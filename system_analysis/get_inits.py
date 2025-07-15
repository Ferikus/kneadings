import numpy as np
import eq_finder.systems_fun as sf
import scipy

from system_analysis.system import DIM, reduced_rhs, rhs_jac

# 1. найти начальные точки для всей сетки параметров
#    1.1 реализовать поиск седло-фокусов
#        c проверкой на нахождение внутри тетраэдра
#    1.2 протянуть его по параметрам
#    1.3 для всех существующих узлов найти начальные точки
#    1.4 организовать передачу массива нач точек в код с нидингами
# 2. нарисовать временные реализации рядом с особым фокусом
# 3. подстроить функцию интегратора под свою систему
#    (с зависимостью от координат с.р.?)


def find_equilibrium(params, initial_guess=None, tol=1e-10, max_iter=100):
    """Находит состояние равновесия системы для заданных параметров."""
    if len(params) == 2:
        alpha, beta = params
        r = 1.0
    else:
        alpha, beta, r = params

    full_params = (alpha, beta, r)

    def func(psi):
        return np.array(rhs(psi, full_params))

    def fprime(psi):
        return rhs_jac(psi, full_params)

    if initial_guess is None:
        initial_guess = np.zeros(3)
    else:
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

    if np.linalg.norm(func(equilibrium)) < tol:
        eigvals, eigvecs = np.linalg.eig(rhs_jac(equilibrium, full_params))
        eq_obj = sf.Equilibrium(equilibrium, eigvals, eigvecs)
        if sf.is3DSaddleFocusWith1dU(eq_obj, sf.STD_PRECISION):
            return equilibrium
    return None


def explore_parameter_grid(start_params, start_eq_coords,
                           up_n, down_n, left_n, right_n,
                           up_step, down_step, left_step, right_step):
    """Обходит сетку параметров и ищет состояния равновесия типа седло-фокус."""
    rows = up_n + down_n + 1
    cols = left_n + right_n + 1
    grid = [[{'params': None, 'equilibrium': None, 'init_pt': None} for _ in range(cols)] for _ in range(rows)]

    start_row, start_col = down_n, left_n
    grid[start_row][start_col]['params'] = start_params
    grid[start_row][start_col]['equilibrium'] = start_eq_coords

    # список всех возможных смещений
    deltas = []
    for di in range(-down_n, up_n + 1):
        for dj in range(-left_n, right_n + 1):
            if (di, dj) == (0, 0):
                continue
            deltas.append((di, dj))

    # сортируем по удалению от центра (сначала ближайшие точки)
    deltas.sort(key=lambda x: abs(x[0]) + abs(x[1]))

    for di, dj in deltas:
        current_i = start_row + di
        current_j = start_col + dj

        if 0 <= current_i < rows and 0 <= current_j < cols:
            delta_alpha = di * (up_step if di > 0 else down_step)
            delta_beta = dj * (right_step if dj > 0 else left_step)

            current_a = start_params[0] + delta_alpha
            current_b = start_params[1] + delta_beta
            current_params = (current_a, current_b, r)
            grid[current_i][current_j]['params'] = current_params

            neighbors = []
            for ni, nj in [(current_i - 1, current_j), (current_i + 1, current_j),
                           (current_i, current_j - 1), (current_i, current_j + 1)]:
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj]['equilibrium'] is not None:
                    neighbors.append(grid[ni][nj]['equilibrium'])

            equilibrium = None
            for guess in neighbors:
                equilibrium = find_equilibrium(current_params, initial_guess=guess)
                if equilibrium is not None:
                    break

            if equilibrium is not None:
                grid[current_i][current_j]['equilibrium'] = equilibrium

                eigvals, eigvecs = np.linalg.eig(rhs_jac(equilibrium, current_params))
                eq_obj = sf.Equilibrium(equilibrium, eigvals, eigvecs)

                def separation_condition(pt, eq_pt):
                    """Выбираем обе сепаратрисы"""
                    return True

                init_pt = sf.getInitPointsOnUnstable1DSeparatrix(eq_obj, separation_condition, sf.STD_PRECISION)[0]
                # берём pt1 = (eq.coordinates + unstVector * ps.separatrixShift).real
                grid[current_i][current_j]['init_pt'] = init_pt

                print(
                    f"Node ({current_i}, {current_j}) | Saddle-focus {equilibrium.round(4)} was found "
                    f"with parameters ({current_a:.3f}, {current_b:.3f}, {r:.3f})")
            else:
                print(
                    f"Node ({current_i}, {current_j}) | No saddle-focus was found "
                    f"with parameters({current_a:.3f}, {current_b:.3f}, {r:.3f})")

    return grid


if __name__ == '__main__':
    # начальные параметры
    w = 0
    a = -2.911209192326542
    b = -1.612684228842761
    r = 1.0

    def rhs_wrapper(psi):
        return rhs(psi, (w, a, b, r))

    def rhs_jac_wrapper(psi):
        return rhs_jac(psi, (w, a, b, r))


    # поиск стартового седло-фокуса
    bounds = [(-0.1, 2 * np.pi + 0.1)] * 3
    borders = [(-1e-15, 2 * np.pi + 1e-15)] * 3

    equilibria = sf.findEquilibria(rhs_wrapper, rhs_jac_wrapper, rhs_wrapper, rhs_jac_wrapper,
                                   lambda phi: phi, bounds, borders,
                                   sf.ShgoEqFinder(1000, 1, 1e-10),
                                   sf.STD_PRECISION)

    start_eq = None
    for eq in equilibria:
        if sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION):
            start_eq = np.array(eq.coordinates)
            print(f"Starting with saddle-focus {start_eq.round(4)} with parameters ({a:.3f}, {b:.3f}, {r:.3f})")
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
        result_grid = explore_parameter_grid((w, a, b, r), start_eq,
                                             up_n, down_n, left_n, right_n,
                                             up_step, down_step, left_step, right_step)
    else:
        print("Start saddle-focus was not found")

    inits = np.empty(DIM * (left_n + right_n + 1) * (up_n + down_n + 1))
    for i in range(left_n + right_n + 1):
        for j in range(up_n + down_n + 1):
            index = (i + j * (left_n + right_n + 1)) * DIM
            if result_grid[j][i]['init_pt'] is not None: # ??? Почему выдаёт исключение. Если нельзя записать None, выбрать большую константу
                inits[index] = result_grid[j][i]['init_pt'][0] # Поменяли местами i и j, так как  result_grid[строка][столбец]
                inits[index + 1] = result_grid[j][i]['init_pt'][1]
                inits[index + 2] = result_grid[j][i]['init_pt'][2]
    print(inits)
