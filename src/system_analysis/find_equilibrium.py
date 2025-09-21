import numpy as np
import lib.eq_finder.systems_fun as sf
import lib.eq_finder.SystOsscills as so

if __name__ == "__main__":
    a = -2.911209192326542
    b = -1.612684228842761
    w = 0
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
            print(f"{start_eq} with parameters ({w:.3f}, {a:.3f}, {b:.3f}, {r:.3f})")