import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib/computation_template'))

import src.computing.workers_kneadings_fbpo as wrk
import src.computing.engines_kneadings_fbpo as engine

from lib.computation_template.engine import workflow, getConfiguration, parseArguments
from src.computing.engines_kneadings_fbpo import get_kneadings_data, check_config_correspondence
from src.routing.route_tools_sepbif import views, map_out_sepbif_route_on_kneadings_set
from src.routing.route_tools_attr import map_out_attr_route_on_kneadings_set

ENGINE_REGISTRY = {'kneadings_fbpo': engine.general_engine}

if __name__ == "__main__":
    parseArguments(sys.argv)
    config = getConfiguration(sys.argv[1])
    task_name = config['task']

    if task_name == 'kneadings_fbpo':
        init_func = wrk.registry['init'][task_name]
        worker = wrk.registry['worker'][task_name]
        engine = ENGINE_REGISTRY[task_name]
        post_process = wrk.registry['post'][task_name]
        def grid_maker(configDict): pass
        workflow(config, init_func, grid_maker, worker, engine, post_process)

    elif task_name == 'route':
        kneadings_data, _, _, kneadings_config = get_kneadings_data(config['kneadings_fbpo']['input_data'])
        check_config_correspondence(kneadings_config, config, ('sf_grid', 'kneadings_fbpo'))

        print("1. Analyze attractors on the route")
        print("2. Analyze separatrix bifurcations on the route")
        selected = int(input())

        if selected == 1:
            map_out_attr_route_on_kneadings_set(config, kneadings_data, views)
        elif selected == 2:
            map_out_sepbif_route_on_kneadings_set(config, kneadings_data, views)
        else:
            raise ValueError("Please select an option from the list")

    else:
        raise ValueError("Please select the task in the config")


