import numpy as np
import matplotlib.pyplot as plt
from experiments import *

class CustomSystem:

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def setParams(self, paramDict):
        for key in paramDict:
            if hasattr(self, key):
                setattr(self, key, paramDict[key])
            else:
                raise KeyError(f"System has no parameter '{key}'")

    def getSystem(self, t, Y):
        y0, y1, y2 = Y
        dydt0 = y1
        dydt1 = y2
        dydt2 = -self.b * y2 - y1 + self.a * y0 - self.a * (y0 ** 3)
        return [dydt0, dydt1, dydt2]

# def sqDistToOrigin(X):
#     x, y = X
#     return x**2 + y**2
#
# def sqDistToCircle(X):
#     x, y = X
#     return np.log10(1e-40 + abs(1. - x**2-y**2))

# sqDistObs = NamedObservable(sqDistToOrigin, "originDistSquared")
# logCircleDistObs = NamedObservable(sqDistToCircle, "circleLogDist")

if __name__ == "__main__":
    curSys = CustomSystem(0.5, 0.5)
    curRhs = curSys.getSystem
    coordNames = ['x', 'y', 'z']

    # tSkip = 1.
    tSkip = 5000.
    tAttr = 20.

    # initPt = np.array([1e-8, 0.0, 0.0])

    initPt = np.array([1e-8, 0.0, 0.0])
    sol1 = computeTrajectory(curRhs, initPt, tSkip, tAttr, {'rtol': 1e-10, 'atol': 1e-10})
    df1 = makeNamedDataframeFromSolution(sol1, coordNames)
    traj = ColoredContinuousDataset(df1)
    pts = ColoredDiscreteDataset(df1.iloc[[0, -1]], colorInfo=['green', 'red'])

    # Построение графиков x(t), y(t), z(t)
    coordLabels = {'x': r'$x(t)$',
                   'y': r'$y(t)$',
                   'z': r'$z(t)$',
                   't': 't'}

    plotParams = {'label': {},
                  'title': {'label': '', 'fontsize': 20}}

    print("PLOT")
    plotDataWithLayout([traj, pts], [[('t', 'x')], [('t', 'y')], [('t', 'z')]],
                       coordLabels, 'trajectory_plot.png', plotParams)

    # Проекции фазового пространства
    print("PLOT XY")
    plotDataWithLayout([traj, pts], [[('x', 'y')]],
                       coordLabels, 'trajectory_xy_plot.png', plotParams)

    print("PLOT YZ")
    plotDataWithLayout([traj, pts], [[('y', 'z')]],
                       coordLabels, 'trajectory_yz_plot.png', plotParams)

    print("PLOT XZ")
    plotDataWithLayout([traj, pts], [[('x', 'z')]],
                       coordLabels, 'trajectory_xz_plot.png', plotParams)





