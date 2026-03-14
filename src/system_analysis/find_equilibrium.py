import numpy as np
import scipy

import lib.eq_finder.systems_fun as sf
import lib.eq_finder.SystOsscills as so

np.set_printoptions(precision=15)


def find_equilibrium_by_guess(rhs, jac, initial_guess, tol=1e-12):
    """Находит состояние равновесия системы для заданных параметров."""
    if initial_guess is None:
        return initial_guess

    initial_guess = np.asarray(initial_guess)

    # Метод HYBR
    result = scipy.optimize.root(
        rhs,
        initial_guess,
        jac=jac,
        method='hybr',
        options={'xtol': tol,     # изменение решения между итерациями
                 'factor': 0.01,  # параметр для начального шага маленький, предотвращает прыжки к другим решениям
                 'maxfev': 1000,  # максимальное число вычислений функции, достаточно для сходимости из близкой точки
                 'diag': None     # без масштабирования для протягивания
        }
    )

    # Метод LM (Левенберг-Марквардт)
    # result = scipy.optimize.root(
    #     rhs,
    #     initial_guess,
    #     jac=jac,
    #     method='lm',
    #     options={'ftol': tol,      # невязка
    #              'xtol': tol,      # изменение решения между итерациями
    #              'gtol': tol,      # градиент
    #              'factor': 0.001,  # параметр для начального шага маленький, предотвращает прыжки к другим решениям
    #              'diag': None,     # без масштабирования для протягивания
    #              'maxiter': 1000}
    # )

    if not result.success:
        return None

    eq_coords = result.x
    eq_obj = sf.getEquilibriumInfo(eq_coords, jac)

    return eq_obj


def correct_equilibrium_coords(rhs, jac, initial_guess, tol=1e-12):
    inner_sf = find_equilibrium_by_guess(rhs, jac, initial_guess, tol)
    if inner_sf is not None:
        return inner_sf.coordinates
    else:
        # raise ValueError("No equilibrium found")
        print(f"Warning: Couldn't find equilibrium by given guess {initial_guess}. The guess itself will be used.")
        return initial_guess


if __name__ == "__main__":
    w = 0.0
    a = -2.6
    b = -1.9
    r = 1.0

    start_sys = so.FourBiharmonicPhaseOscillators(w, a, b, r)

    bounds = [(-0.1, 2 * np.pi + 0.1)] * 2
    borders = [(-1e-15, 2 * np.pi + 1e-15)] * 2

    # первые две функции -- общая система, вторые две -- в которой ищем с.р., дальше функция приведения
    equilibria = sf.findEquilibria(lambda psis: start_sys.getReducedSystem(psis),
                                   lambda psis: start_sys.getReducedSystemJac(psis),
                                   lambda psis: start_sys.getRestriction(psis),
                                   lambda psis: start_sys.getRestrictionJac(psis),
                                   lambda phi: np.concatenate([[0.], phi]), bounds, borders,
                                   sf.ShgoEqFinder(1000, 1, 1e-10),
                                   sf.STD_PRECISION)

    start_eq = None
    for eq in equilibria:  # перебираем все с.р., которые были найдены
        if sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION):
            start_eq = np.array(eq.coordinates)
            print(f"{start_eq} with parameters ({w:.3f}, {a:.15f}, {b:.15f}, {r:.3f})")