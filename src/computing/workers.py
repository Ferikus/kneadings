registry = {
    "worker": {},
    "init": {},
    "post": {}
}

from src.computing.workers_kneadings import init_kneadings_fbpo, worker_kneadings_fbpo, post_kneadings_fbpo  # kneadings task
from src.computing.workers_periodicity import worker_periodicity_fbpo, post_periodicity_fbpo  # periodicity task
from src.computing.workers_symmetry_detectives import worker_symmetry_detectives, post_symmetry_detectives  # symmetry detectives task
