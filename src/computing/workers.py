from lib.computation_template.workers_utils import register
from src.computing.engines import get_data, get_config_data, check_config_correspondence

registry = {
    "worker": {},
    "init": {},
    "post": {}
}


def make_kneadings_based_worker(task_name, sweep):
    @register(registry, 'worker', task_name)
    def kneadings_based_worker(config, initResult, timeStamp):
        def_sys_dict = config['defaultSystem']
        w = def_sys_dict['w']
        a = def_sys_dict['a']
        b = def_sys_dict['b']
        r = def_sys_dict['r']
        param_to_index = def_sys_dict['param_to_index']

        grid_dict = config['grid']
        left_n = grid_dict['first']['left_n']
        right_n = grid_dict['first']['right_n']
        up_n = grid_dict['second']['up_n']
        down_n = grid_dict['second']['down_n']
        param_x_name = grid_dict['first']['name']
        param_y_name = grid_dict['second']['name']

        task_dict = config[task_name]
        dt = task_dict['dt']
        n = task_dict['n']
        stride = task_dict['stride']
        kneadings_start = task_dict['kneadings_start']
        kneadings_end = task_dict['kneadings_end']
        input_data_path = task_dict['input_data']

        inits = initResult['inits']
        nones = initResult['nones']
        params_x = initResult['params_x']
        params_y = initResult['params_y']
        inner_sf_set = initResult['inner_sf_set']

        if input_data_path is not None:
            prev_config = get_config_data(input_data_path)
            check_config_correspondence(prev_config, config, ('sf_grid', ), check_source=True)
            task_data = get_data(input_data_path, config)
            _, _, _, _, task_core_data = task_data
        else:
            def_params = [w, a, b, r]
            task_core_data = sweep(
                inits,
                nones,
                params_x,
                params_y,
                def_params,
                param_to_index,
                param_x_name,
                param_y_name,
                up_n,
                down_n,
                left_n,
                right_n,
                dt,
                n,
                stride,
                kneadings_start,
                kneadings_end,
                inner_sf_set
            )

        return {'task_core_data': task_core_data}

    return kneadings_based_worker


# goes after registry and worker template
from src.computing.workers_kneadings import init_kneadings_fbpo, worker_kneadings_fbpo, post_kneadings_fbpo  # kneadings task
from src.computing.workers_regularity import worker_regularity, post_regularity  # regularity task
from src.computing.workers_complexity import worker_complexity, post_complexity  # complexity task
from src.computing.workers_periodicity import worker_periodicity, post_periodicity  # periodicity task
from src.computing.workers_symmetry_detectives import worker_symmetry_detectives, post_symmetry_detectives  # symmetry detectives task
from src.computing.workers_route import init_route, worker_route, post_route  # route task