import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib/computation_template'))

import lib.computation_template.workers_sl as wrk
import src.computing.workers_kneadings_fbpo  # нужно для регистрации воркеров
from src.computing.engines_kneadings_fbpo import ENGINE_REGISTRY
from lib.computation_template.engine import workflow, getConfiguration, parseArguments


if __name__ == "__main__":
    parseArguments(sys.argv)
    configDict = getConfiguration(sys.argv[1])
    taskName = configDict['task']
    initFunc = wrk.registry['init'][taskName]
    worker = wrk.registry['worker'][taskName]
    engine = ENGINE_REGISTRY[taskName]
    postProcess = wrk.registry['post'][taskName]
    workflow(configDict, initFunc, worker, engine, postProcess)

