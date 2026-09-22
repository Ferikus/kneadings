import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib/computation_template'))

import src.computing.workers as wrk
import src.computing.engines as engine

from lib.computation_template.engine import workflow, getConfiguration, parseArguments

ENGINE_REGISTRY = {'kneadings': engine.general_engine,
                   'regularity': engine.general_engine,
                   'complexity': engine.general_engine,
                   'periodicity': engine.general_engine,
                   'symmetry_detectives': engine.general_engine,
                   'route': engine.general_engine}

if __name__ == "__main__":
    parseArguments(sys.argv)
    config = getConfiguration(sys.argv[1])
    task_name = config['task']

    init_func = wrk.registry['init'][task_name]
    worker = wrk.registry['worker'][task_name]
    engine = ENGINE_REGISTRY[task_name]
    post_process = wrk.registry['post'][task_name]
    def grid_maker(configDict): pass
    workflow(config, init_func, grid_maker, worker, engine, post_process)



