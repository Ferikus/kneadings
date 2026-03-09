import numpy as np

import lib.eq_finder.SystOsscills as so
from src.system_analysis.taskutils import get_domain_np, t
from src.system_analysis.find_equilibrium import find_equilibrium_by_guess


def get_plane_coeffs(pt1, pt2, pt3):
    v1 = pt2 - pt1
    v2 = pt3 - pt2

    x1, y1, z1 = v1
    x2, y2, z2 = v2

    a = y1 * z2 - z1 * y2
    b = -(x1 * z2 - z1 * x2)
    c = x1 * y2 - y1 * x2

    d = np.array([a, b, c]) @ np.array(-pt1)

    return a, b, c, d


def get_poincare_section_coeffs_by_domain(inner_sf, domain_num):
    pt1 = np.array([0.5 * np.pi, 1.0 * np.pi, 1.5 * np.pi])  # pt_w
    pt2 = np.array([1.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi])  # (pt_b + pt_c) / 2

    for i in range(domain_num):
        inner_sf = np.array(t(inner_sf))
        pt2 = np.array(t(pt2))

    coeffs = get_plane_coeffs(pt1, pt2, inner_sf)

    if domain_num in [0, 2]:
        coeffs = [-coeff for coeff in coeffs]

    return coeffs


def get_poincare_section_coeffs(inner_sf):
    if inner_sf is None:
        default_coeffs = [1.04187071692464, -0.242406134831432, -0.557058447261778, 1.75005072553774,
                          -0.242406134831432, -0.557058447261777, -0.242406134831432, 3.27313339028078,
                          -0.557058447261779, -0.242406134831431, 1.04187071692464, -3.27313339028078,
                          -0.242406134831431, 1.04187071692464, -0.242406134831431, -1.75005072553774]
        return default_coeffs

    curr_domain = get_domain_np(inner_sf)
    assert curr_domain == 0, "Inner saddle-focus is out of the 0 domain"

    all_coeffs = np.zeros(16)
    for domain_num in range(4):
        coeffs = get_poincare_section_coeffs_by_domain(inner_sf, domain_num)
        for i in range(4):
            all_coeffs[domain_num * 4 + i] = coeffs[i]

    return all_coeffs


if __name__ == '__main__':
    params = [0.0, -2.911209192326542, -1.612684228842761, 1.0]
    sys = so.FourBiharmonicPhaseOscillators(*params)
    rhs = sys.getReducedSystem
    jac = sys.getReducedSystemJac
    guess = np.array([1.427257804280822, 3.2091500304528755, 4.414529919493724])
    inner_sf = find_equilibrium_by_guess(rhs, jac, guess).coordinates
    all_coeffs = get_poincare_section_coeffs(inner_sf)
    print(all_coeffs)
