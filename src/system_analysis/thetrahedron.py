import numpy as np

# Вычисляем "наблюдаемую" - сумму расстояний до граней тетраэдра
ptO = np.array([0, 0, 0.])
ptA = np.array([0, 0, 2*np.pi])
ptB = np.array([0, 2*np.pi, 2*np.pi])
ptC = np.array([2*np.pi, 2*np.pi, 2*np.pi])
ptW = (ptO + ptA + ptB + ptC)/4.

tetra = [ptO, ptA, ptB, ptO, ptC, ptA, ptB, ptC]
tetraX, tetraY, tetraZ = np.array(tetra).T

frame = [ptW, ptO, ptW, ptA, ptW, ptB, ptW, ptC]
frameX, frameY, frameZ = np.array(frame).T

plank = np.array([(0, np.pi, np.pi), ptW, (np.pi, np.pi, 2*np.pi)])
plankX, plankY, plankZ = plank.T