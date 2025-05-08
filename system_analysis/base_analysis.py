# 1. найти начальные точки для всей сетки параметров
#   - найти координаты состояний равновесия
#   - отсеять всё, кроме седло-фокусов
#   - найти начальные точки сепаратрис
# 2. нарисовать по ним временные реализации
# 3. подстроить функцию интегратора под свою систему

import numpy as np
import eq_finder.systems_fun as sf


# def rhs(phi, params):
#     alpha, beta, omega, r = params
#     dphi = [omega] * 4
#     for i in range(4):
#         for j in range(4):
#             dphi[i] += 1 / 4 * (-np.sin(phi[i] - phi[j] + alpha) + r * np.sin(2 * (phi[i] - phi[j]) + beta))
#     return dphi
#
#
# def rhs_jac(phi, params):
#     alpha, beta, omega, r = params
#     jac = np.zeros((4, 4))
#
#     for i in range(4):
#         for j in range(4):
#             deriv = 0.0
#             if i == j:
#                 for k in range(4):
#                     deriv += 1 / 4 * (-np.cos(phi[i] - phi[k] + alpha) + 2 * r * np.cos(2 * (phi[i] - phi[k]) + beta))
#             else:
#                 deriv = 1 / 4 * (np.cos(phi[i] - phi[j] + alpha) - 2 * r * np.cos(2 * (phi[i] - phi[j]) + beta))
#             jac[i, j] = deriv
#     return jac


def link_function(phi, params):
    alpha, beta, r = params
    return -np.sin(phi + alpha) + r * np.sin(2 * phi + beta)


def link_function_deriv(phi, params):
    alpha, beta, r = params
    return -np.cos(phi + alpha) + 2 * r * np.cos(2 * phi + beta)


def rhs(psi, params):
    alpha, beta, r = params
    dpsi = [0.] * 3
    for i in range(3):
        for j in range(3):
            dpsi[i] += 1 / 4 * (link_function(psi[i] - psi[j], params) - link_function(-psi[j], params))
    return dpsi


def rhs_jac(psi, params):
    alpha, beta, r = params
    jac = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            deriv = 0.0
            if i == j:
                for k in range(3):
                    deriv = 1 / 4 * (link_function_deriv(psi[i] - psi[j], params) - link_function_deriv(-psi[j], params))
            else:
                deriv = - 1 / 4 * (link_function_deriv(psi[i] - psi[j], params) + link_function_deriv(-psi[j], params))
            jac[i, j] = deriv
    return jac


alpha = -2.911209192326542
beta = -1.612684228842761
r = 1
params = [alpha, beta, r]

bounds = [(-0.1, 2 * np.pi + 0.1), (-0.1, 2 * np.pi + 0.1), (-0.1, 2 * np.pi + 0.1)]
borders = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

rhs_current = lambda phi: rhs(phi, params)
rhs_jac_current = lambda phi: rhs_jac(phi, params)
res = sf.findEquilibria(rhs_current, rhs_jac_current, rhs_current, rhs_jac_current,
                     lambda phi: phi, bounds, borders, sf.ShgoEqFinder(1000, 1, 1e-10), sf.STD_PRECISION)
# print(res)
for eq in res:
    if sf.is3DSaddleFocusWith1dU(eq, sf.STD_PRECISION):
        print(eq.coordinates)