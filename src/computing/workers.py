registry = {
    "worker": {},
    "init": {},
    "post": {}
}

from src.computing.workers_kneadings import init_kneadings_fbpo, worker_kneadings_fbpo, post_kneadings_fbpo  # kneadings task
from src.computing.workers_regularity import worker_regularity, post_regularity  # regularity task
from src.computing.workers_complexity import worker_complexity, post_complexity  # complexity task
from src.computing.workers_periodicity import worker_periodicity, post_periodicity  # periodicity task
from src.computing.workers_symmetry_detectives import worker_symmetry_detectives, post_symmetry_detectives  # symmetry detectives task
from src.computing.workers_route import init_route, worker_route, post_route  # route task
