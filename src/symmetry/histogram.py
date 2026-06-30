import numpy as np

from src.plotting.plot_attractors import get_kneadings_trajectory


def getGridIndex(ptCoord, nLevels):
    # print(f"{ptCoord = }")
    # preliminary stuff
    ptV1 = np.array([0., 0, 0])
    ptV2 = np.array([0, 0, 2 * np.pi])
    ptV3 = np.array([0, 2 * np.pi, 2 * np.pi])
    ptV4 = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi])
    # ptV1 -> ptV2 -> ptV3 -> ptV4
    ptW = 0.25 * (ptV1 + ptV2 + ptV3 + ptV4)
    w1 = ptV1 - ptW
    w2 = ptV2 - ptW
    w3 = ptV3 - ptW
    w4 = ptV4 - ptW
    matBary = np.vstack((np.column_stack((w1, w2, w3, w4)), [1., 1, 1, 1]))
    baryCoords = np.linalg.solve(matBary, np.hstack((ptCoord - ptW, 1)))
    # figuring domain
    domainIndex = np.argmin(baryCoords)
    # and we don't need to go to zero-domain and re-expand anything;
    # I'm just repeating the argument above to which tetrahedron point belongs
    c1 = baryCoords[(domainIndex + 1) % 4] - baryCoords[domainIndex]
    c2 = baryCoords[(domainIndex + 2) % 4] - baryCoords[domainIndex]
    c3 = baryCoords[(domainIndex + 3) % 4] - baryCoords[domainIndex]
    coeffs = np.array([c1, c2, c3])
    # the rest is exactly the same as before
    discrCoeffs = (np.floor(nLevels * coeffs)).astype(int)
    rst = tuple([(int(domainIndex) + 1) % 4] + [int(d) for d in discrCoeffs])
    return rst


def getSymmetryTypeByHistogram(dat, nLevels, precision=0.1):
    histogram = np.zeros((4, nLevels, nLevels, nLevels), dtype=int)

    for pt in dat:
        gridInd = getGridIndex(pt, nLevels)
        histogram[gridInd] += 1

    histogram = histogram / len(dat)

    nonEmptyDomainInd = getGridIndex(dat[0], nLevels)[0]

    symmDistT1 = float(np.max(np.abs(histogram[nonEmptyDomainInd] - histogram[(nonEmptyDomainInd + 1) % 4])))  # if zero, then T1 symmetry
    symmDistT2 = float(np.max(np.abs(histogram[nonEmptyDomainInd] - histogram[(nonEmptyDomainInd + 2) % 4])))  # if zero, then T2 symmetry
    # if both zero, then assymetrical (T4 symmetry)

    # print(symmDistT1, symmDistT2)

    symmType = 0
    if symmDistT1 < precision and symmDistT2 < precision:
        symmType = 4
    elif symmDistT1 < precision:
        symmType = 1
    elif symmDistT2 < precision:
        symmType = 2

    return symmType, symmDistT1, symmDistT2


if __name__ == "__main__":
    params = [0.0, -2.487, -1.612684228842761, 1.0]
    _, dat, _, _, _ = get_kneadings_trajectory(params, dt=0.01, n=200000, stride=1, kneadings_start=0, kneadings_end=7, debug=False)

    nLevels = 256
    symmType = getSymmetryTypeByHistogram(dat, nLevels, precision=0.1)
    print(symmType)