import numpy as np

DIM = 3


def link_function(phi, params):
    """Вспомогательная функция связи"""
    _, a, b, r = params
    return -np.sin(phi + a) + r * np.sin(2 * phi + b)


def link_function_deriv(phi, params):
    """Производная вспомогательной функции связи"""
    _, a, b, r = params
    return -np.cos(phi + a) + 2 * r * np.cos(2 * phi + b)


def full_rhs(phis, params):
    """Полная система четырёх идентичных глобально связанных фазовых осцилляторов с бигармонической связью"""
    w, a, b, r = params
    rhs_phis = [w] * 4
    for i in range(4):
        for j in range(4):
            rhs_phis[i] += 0.25 * link_function(phis[i] - phis[j], params)
    return rhs_phis

def reduced_rhs(psis, params):
    """Редуцированная система четырёх идентичных глобально связанных фазовых осцилляторов с бигармонической связью"""
    # rhs_psis = list(rhs_psis)
    phis = [0.] + psis
    rhs_phis = full_rhs(phis, params)
    rhs_psis = [0.] * 4
    for i in range(4):
        rhs_psis[i] = rhs_phis[i] - rhs_phis[0]
    return rhs_psis[1:]


def rhs_jac(psi, params):
    """Якобиан системы"""
    jac = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            deriv = 0.0
            if i == j:
                for k in range(3):
                    deriv = 1 / 4 * (
                            link_function_deriv(psi[i] - psi[j], params) - link_function_deriv(-psi[j], params))
            else:
                deriv = - 1 / 4 * (link_function_deriv(psi[i] - psi[j], params) + link_function_deriv(-psi[j], params))
            jac[i, j] = deriv
    return jac