"""Utilities for managing gaguge groups for global model paper experiments."""

import glob
import pathlib
import tqdm

from backend import data_paths


# --- Utilities for Loading Gauge Groups ---------------------------------------


def load_gauge_group(
    gauge_group_path: pathlib.Path
) -> list[str]:
    """Loads and gauge group file."""

    with open(gauge_group_path, 'rt') as f:
        lines = f.readlines()

    gauges = [gauge.strip('\n') for gauge in lines]

    return gauges


def write_gauge_group(filename: str, basins: list[str]):
    with open(filename, 'wt') as f:
        for basin in basins:
            f.write("%s\n" % basin)

def get_full_gauge_group() -> list[str]:
    """Returns the gauge group will all gauges used for the paper."""
    return load_gauge_group(data_paths.FULL_GAUGE_GROUP_FILE)


def load_experiment_gauge_groups(
    experiment_dir_name: str
) -> dict[str, list[str]]:
    """Loads a dictionary of gauge groups for a split-sample experiment."""

    gauge_groups_dir = data_paths.GAUGE_GROUPS_DIR / experiment_dir_name
    gauge_group_paths = glob.glob(str(gauge_groups_dir / '*'))

    print(f'Working on {experiment_dir_name} ...')

    experiment_gauge_groups = {}
    for path in tqdm.tqdm(gauge_group_paths):
        split_name = pathlib.Path(path).stem
        experiment_gauge_groups[split_name] = load_gauge_group(
            gauge_group_path=path)

    return experiment_gauge_groups


def load_all_ungauged_gauge_groups() -> dict[str, dict[str, list[str]]]:
    """Loads gauge groups for all ungauged experiments."""
    return {
        experiment: load_experiment_gauge_groups(
            experiment_dir_name=experiment) 
        for experiment in data_paths.UNGAUGED_EXPERIMENTS
    }


