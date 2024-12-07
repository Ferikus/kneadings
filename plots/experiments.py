from typing import List

from scipy.integrate import solve_ivp
import pandas as pd
import matplotlib.pyplot as plt


def isDictKeysCompatible(refDict, passedDict, msg="Error, keys are incompatible!"):
    assert set(passedDict.keys()) <= set(refDict.keys()), \
        f"{msg}\n{list(passedDict.keys()) = }\nvs\n{list(refDict.keys()) = }"


def computeTrajectory(sysRhs, initPt, tSkip, tAttractor, odeParams):
    defaultOdeParams = {'method': 'RK45', 'rtol': 1e-8, 'atol': 1e-8}
    isDictKeysCompatible(defaultOdeParams, odeParams, )
    defaultOdeParams.update(odeParams)

    startPt = initPt
    if tSkip is not None:
        sol = solve_ivp(sysRhs, [0, tSkip], startPt, **defaultOdeParams)
        startPt = sol.y[:, -1]

    # print(f"{startPt = }")
    sol = solve_ivp(sysRhs, [0, tAttractor], startPt, **defaultOdeParams)

    return sol


def makeNamedDataframeFromSolution(odeSolution, phaseVarNames, fromEvents=False):
    if fromEvents:
        xs, ts = odeSolution.y_events, odeSolution.t_events
    else:
        xs, ts = odeSolution.y, odeSolution.t

    dataDict = {colName: x for colName, x in zip(phaseVarNames, xs)}
    dataDict['t'] = ts
    df = pd.DataFrame(dataDict)

    return df


class NamedObservable:
    def __init__(self, observFunc, observName):
        self.observFunc = observFunc
        self.observName = observName

    def __call__(self, X):
        return self.observFunc(X)


def calculateColumnsForObservables(dataFrame: pd.DataFrame, phaseVarNames, observables: List[NamedObservable]):
    for obs in observables:
        dataFrame[obs.observName] = dataFrame[phaseVarNames].apply(obs.observFunc, axis=1)

    return dataFrame


class ColoredContinuousDataset:
    defaultContinuousDatasetProperties = {'color': 'black', 'alpha': 1., 'marker': None,
                                          'linestyle': '-', 'linewidth': 1}

    def __init__(self, data: pd.DataFrame, label=None, **kwargs, ):
        self.data = data
        self.label = label
        dcdp = ColoredContinuousDataset.defaultContinuousDatasetProperties
        isDictKeysCompatible(dcdp, kwargs)
        self.plotArgs = dcdp.copy() | kwargs


class ColoredDiscreteDataset:
    defaultDiscreteDatasetProperties = {'alpha': 1.}

    # markerInfo    й can be only a single string: list of markers are not supported
    # https://stackoverflow.com/questions/18800944/changing-marker-style-in-scatter-plot-according-to-third-variable
    def __init__(self, data: pd.DataFrame, colorInfo=None, sizeInfo=None, markerInfo=None, label=None, **kwargs):
        self.data = data
        self.colorInfo = colorInfo
        self.sizeInfo = sizeInfo
        self.markerInfo = markerInfo
        self.label = label
        dddp = ColoredDiscreteDataset.defaultDiscreteDatasetProperties
        isDictKeysCompatible(dddp, kwargs)
        self.scatterArgs = kwargs


def plotDataWithLayout(coloredData: List[ColoredContinuousDataset | ColoredDiscreteDataset],
                       layout, coordLabels, pathToOutImage, plotParams=None, showLegend=False):
    plotParams = {} if plotParams is None else plotParams

    # TODO: how to make non-rectangular layouts
    nRows = len(layout)
    nCols = max([len(lt) for lt in layout])
    print(f"{nRows = } {nCols = }")
    # assert min([len(L) for L in layout]) == nCols and \
    #        max([len(L) for L in layout]) == nCols, "The layout is not rectangular!"

    fig = plt.figure(layout='constrained')
    # fig, axes = plt.subplots(nRows, nCols,
    #                          #sharex=True,
    #                          #sharey=True,
    #                          layout='constrained')
    # устанавливаем большой заголовок
    titleParams = plotParams.get('title', {})
    titleLabel = titleParams.get('label', None)
    if titleLabel is not None:
        fig.suptitle(titleLabel, **titleParams)

    for i, row in enumerate(layout):
        for j, varNames in enumerate(row):
            xvar, yvar = varNames
            # ax = axes[i][j]
            print(f"{i = }")
            print(f"{j = }")
            ind = i*nCols + j + 1
            print(f"{ind = }")
            print("###")
            ax = fig.add_subplot(nRows, nCols, ind)

            # устанавливаем метки и свойства осей
            xlabel = coordLabels[xvar]
            ylabel = coordLabels[yvar]
            labelParams = plotParams.get('label', {})
            ax.set_xlabel(xlabel, **labelParams)
            ax.set_ylabel(ylabel, **labelParams)

            # plot data
            for cd in coloredData:
                if type(cd) == ColoredContinuousDataset:
                    # https://stackoverflow.com/a/65163984
                    # explicitly pass empty format string in order
                    # to suppress the warning
                    ax.plot(xvar, yvar, '', data=cd.data, **cd.plotArgs)
                elif type(cd) == ColoredDiscreteDataset:
                    ax.scatter(xvar, yvar, data=cd.data, c=cd.colorInfo, marker=cd.markerInfo, s=cd.sizeInfo)
                else:
                    pass

    plt.savefig(pathToOutImage, facecolor='white', **plotParams.get('figure', {}))
