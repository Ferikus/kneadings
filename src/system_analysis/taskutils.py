import numpy as np


def det4x4(m):
    det = 0.0
    sign = 1.0

    minor = [0.] * 9

    for col in range(4):
        minor_row_idx = 0
        for i in range(1, 4):
            minor_col_idx = 0
            for j in range(4):
                if j != col:
                    minor[minor_row_idx * 3 + minor_col_idx] = m[i * 4 + j]
                    minor_col_idx += 1
            minor_row_idx += 1

        det_minor = (
            minor[0] * (minor[4] * minor[8] - minor[5] * minor[7]) -
            minor[1] * (minor[3] * minor[8] - minor[5] * minor[6]) +
            minor[2] * (minor[3] * minor[7] - minor[4] * minor[6])
        )

        det += sign * m[0 * 4 + col] * det_minor
        sign *= -1.0
    return det


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


def bary_expansion(pt):
    pt_o = [0.] * 3
    pt_a = [0.] * 3
    pt_b = [0.] * 3
    pt_c = [0.] * 3
    pt_w = [0.] * 3
    vec_wa = [0.] * 3
    vec_wb = [0.] * 3
    vec_wc = [0.] * 3
    vec_wo = [0.] * 3
    mat_bary = [0.] * 16
    rhs = [0.] * 4

    pt_o[0] = 0.0; pt_o[1] = 0.0; pt_o[2] = 0.0
    pt_a[0] = 0.0; pt_a[1] = 0.0; pt_a[2] = 2 * np.pi
    pt_b[0] = 0.0; pt_b[1] = 2 * np.pi; pt_b[2] = 2 * np.pi
    pt_c[0] = 2 * np.pi; pt_c[1] = 2 * np.pi; pt_c[2] = 2 * np.pi

    pt_w[0] = 0.25 * (pt_a[0] + pt_b[0] + pt_c[0] - 3 * pt_o[0])
    pt_w[1] = 0.25 * (pt_a[1] + pt_b[1] + pt_c[1] - 3 * pt_o[1])
    pt_w[2] = 0.25 * (pt_a[2] + pt_b[2] + pt_c[2] - 3 * pt_o[2])

    for i in range(3):
        vec_wa[i] = pt_a[i] - pt_w[i]
        vec_wb[i] = pt_b[i] - pt_w[i]
        vec_wc[i] = pt_c[i] - pt_w[i]
        vec_wo[i] = pt_o[i] - pt_w[i]

        mat_bary[4 * i] = vec_wa[i]
        mat_bary[4 * i + 1] = vec_wb[i]
        mat_bary[4 * i + 2] = vec_wc[i]
        mat_bary[4 * i + 3] = vec_wo[i]

        rhs[i] = pt[i] - pt_w[i]

    mat_bary[12] = 1.; mat_bary[13] = 1.; mat_bary[14] = 1.; mat_bary[15] = 1.
    rhs[3] = 1.

    main_det = det4x4(mat_bary)

    bary_coords = [0.] * 4

    if abs(main_det) < 1e-12:
        bary_coords[:] = 0.
        return bary_coords

    # заполняем координаты решая систему методом Крамера
    for col in range(4):
        modified_mat = mat_bary.copy()
        for row in range(4):
            modified_mat[4 * row + col] = rhs[row]

        coord_det = det4x4(modified_mat)
        bary_coords[col] = coord_det / main_det

    return bary_coords


def get_domain_np(pt):
    bary_expansion = bary_expansion_np(pt)
    return np.argmin(bary_expansion)


def get_domain_num(bary_expansion):
    min_coord = bary_expansion[0]
    i = 0
    domain_num = i
    while i < 4:
        if bary_expansion[i] < min_coord:
            min_coord = bary_expansion[i]
            domain_num = i
        i += 1
    return domain_num


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