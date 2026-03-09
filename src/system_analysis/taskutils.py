import numpy as np


def bary_expansion_np(globalPtCoords):
    """
    globalPtCoords must be a 3d vector with 0 <= x <= y <= z <= 2pi, i.e. inside a CIR
    returns an expansion of (globalPtCoords - center of mass) in barycentric coordinates
    """
    ptO = np.array([0., 0, 0])
    ptA = np.array([0, 0, 2*np.pi])
    ptB = np.array([0, 2*np.pi, 2*np.pi])
    ptC = np.array([2*np.pi, 2*np.pi, 2*np.pi])
    ptW = 0.25 * ((ptA-ptO) + (ptB - ptO) + (ptC - ptO))
    vecWA = ptA - ptW
    vecWB = ptB - ptW
    vecWC = ptC - ptW
    vecWO = ptO - ptW
    matBary = np.vstack((np.column_stack((vecWA, vecWB, vecWC, vecWO)), [1., 1, 1, 1]))
    baryCoords = np.linalg.solve(matBary, np.hstack((globalPtCoords - ptW, 1)))
    return baryCoords


def get_domain_np(pt):
    bary_expansion = bary_expansion_np(pt)
    return np.argmin(bary_expansion)


# def simplexDistance(globalPt):
#     barExp = baryExpansion(globalPt)
#     perDomainCoord = barExp - min(barExp)
#     return sum(barExp)


def avg_face_distance(globalPt):
    x, y, z = globalPt
    return 0.25*(x**2 + (y-x)**2 + (z-y)**2 + (z - 2*np.pi)**2)


def t(pt):
    x, y, z = pt
    return [y-x, z-x, 2*np.pi - x]


def eucl_sq(pt):
    x, y, z = pt
    return np.sqrt(x*x + y*y + z*z)


def splay_distance(pt):
    splay = np.array([np.pi/2, np.pi, 3*np.pi/2])
    return eucl_sq(splay - pt)


def avg_splay_distance(pt):
    splay = np.array([np.pi/2, np.pi, 3*np.pi/2])
    return 0.25*(eucl_sq(splay - pt) + eucl_sq(splay - t(pt)) + eucl_sq(splay-t(t(pt))) + eucl_sq(splay-t(t(t(pt)))))