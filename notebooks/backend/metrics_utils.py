"""Utilities for calculating metrics across model runs."""

import os
import pathlib
import shutil
from typing import Optional

import pandas as pd
import tqdm
import xarray

from backend import data_paths
from backend import loading_utils
from backend import metrics


OBS_VARIABLE = 'observation'
UNNORMALIZED_OBS_VARIABLE = 'unnormalized_observation'
GOOGLE_VARIABLE = 'google_prediction'
GLOFAS_VARIABLE = 'glofas_prediction'

METRICS = [
    'MSE',
    'RMSE',
    'NSE',
    'log-NSE',
    'Beta-NSE',
    'Alpha-NSE',
    'KGE',
    'log-KGE',
    'Pearson-r',
    'Beta-KGE',
    # 'Peak-Timing',
    # 'Missed-Peaks',
    'Peak-MAPE',
    'FLV',
    'FHV',
    'FMS',
]


# --- Save Metrics Results -----------------------------------------------------


def save_metrics_df(
    df: Optional[pd.DataFrame],
    metric: str,
    base_path: pathlib.Path,
    path_modifier: Optional[str] = None,
) -> pathlib.Path:
    """Saves a metrics dataframe and returns the path."""

    # Construct file path.
    path = base_path
    if path_modifier is not None:
        path = path / path_modifier

    # Create the directory if it does not exist.    
    loading_utils.create_remote_folder_if_necessary(path)

    # Path to file.
    filepath = path / f'{metric}.csv'

    # If no dataframe is passed, only return the path.
    if df is None:
        return filepath

    # Create the directory if necessary.
    loading_utils.create_remote_folder_if_necessary(path)

    # Save the dataframe to csv.
    with open(filepath, 'w') as f:
        df.to_csv(f)

    return filepath


def load_metrics_df(
    filepath: pathlib.Path,
) -> Optional[pd.DataFrame]:
    """Loads a metrics dataframe."""

    # If the file does not exist, return None.
    if not os.path.exists(filepath):
        raise ValueError('Metrics file does not exist at path: ', filepath)
#         return None

    # Otherwise load and return.
    with open(filepath, 'r') as f:
        return pd.read_csv(f, index_col='Unnamed: 0')


def load_metrics_for_experiment(
    metric: str,
    experiment: str,
    base_path: pathlib.Path,
):
    filepath = save_metrics_df(
        df=None,
        metric=metric,
        base_path=base_path,
        path_modifier=experiment,
    )
    return load_metrics_df(filepath=filepath)

# --- General Metrics Calculations ---------------------------------------------


def _metrics_for_gauge_time_period_and_lead_time(
    ds: xarray.Dataset,
    gauge: str,
    lead_time: int,
    metrics_to_calculate: list[str],
    sim_variable: str,
    obs_variable: str,
    time_period: Optional[list[str]] = None,
) -> dict[str, float]:
    """Pulls sim and obs & calculate metrics from a properly formed Dataset."""

    # Pull simulation and observations from dataset at a particular lead time.
    obs = ds[obs_variable]
    sim = ds[sim_variable]
  
    # Pull only selected lead time, if available.
    if 'lead_time' in obs.dims:
        obs = obs.sel(lead_time=lead_time)
        obs = obs.drop_vars(['lead_time'])
    if 'lead_time' in sim.dims:
        sim = sim.sel(lead_time=lead_time)
        sim = sim.drop_vars(['lead_time'])

    # Pull gauge.
    if 'gauge_id' in sim.dims:
        sim = sim.sel(gauge_id=gauge)
    if 'gauge_id' in obs.dims:
        obs = obs.sel(gauge_id=gauge)

    # Pull time period.
    if time_period is not None:
        obs = obs.sel(time=slice(*time_period))
        sim = sim.sel(time=slice(*time_period))

    # Drop unnecessary dimensions.
    obs = obs.drop_vars(['gauge_id'])
    sim = sim.drop_vars(['gauge_id'])

    # Calculate metrics for this slice.
    return metrics.calculate_metrics(
        obs=obs,
        sim=sim,
        metrics=metrics_to_calculate,
        datetime_coord='time',
    )


def _metrics_for_one_gauge(
    ds: xarray.Dataset,
    gauge: str,
    sim_variable: str,
    obs_variable: str,
    metrics_to_calculate: list[str],
    lead_times: list[int] = data_paths.LEAD_TIMES,
    time_period: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Loads data and calculates metrics for one gauge."""

    series_at_lead_time = []
    for lead_time in lead_times:

        # Calculate the metrics for a particular lead time.
        metrics_dict = _metrics_for_gauge_time_period_and_lead_time(
            ds=ds,
            gauge=gauge,
            lead_time=lead_time,
            metrics_to_calculate=metrics_to_calculate,
            sim_variable=sim_variable,
            obs_variable=obs_variable,
            time_period=time_period,
        )
    
        # Turn the metrics dictionary into a pandas series, to be concateated later.
        series_at_lead_time.append(
            pd.Series(
                metrics_dict.values(),
                index=metrics_dict.keys(),
                name=lead_time
            )
        )

    return pd.concat(series_at_lead_time, axis=1)


def _calculate_and_save_metrics_for_one_gague(
    ds: xarray.Dataset,
    gauge: str,
    sim_variable: str,
    obs_variable: str,
    base_path: pathlib.Path,
    metrics_to_calculate: Optional[list[str]] = None,
    lead_times: list[str] = data_paths.LEAD_TIMES,
    time_period: Optional[list[str]] = None,
    path_modifier: Optional[str] = None,
) -> pd.DataFrame:
    """Calculates metrics for many gauges."""

    # If a list of metrics is not provided, calculate all metrics.
    if metrics_to_calculate is None:
        metrics_to_calculate = metrics.get_available_metrics()

    # Check if gauge file already exists.
    filepath = save_metrics_df(
        df=None,
        metric=gauge,
        base_path=base_path,
        path_modifier=path_modifier,
    )
    if os.path.exists(filepath):
        return load_metrics_df(filepath=filepath)

    gauge_metrics_df = _metrics_for_one_gauge(
      ds=ds,
      gauge=gauge,
      sim_variable=sim_variable,
      obs_variable=obs_variable,
      metrics_to_calculate=metrics_to_calculate,
      lead_times=lead_times,
      time_period=time_period,
    )
    
    # Save metrics for this gauge.
    _ = save_metrics_df(
        df=gauge_metrics_df,
        metric=gauge,
        base_path=base_path,
        path_modifier=path_modifier,
    )

    return gauge_metrics_df


def calculate_and_save_metrics_for_many_gagues(
    ds: xarray.Dataset,
    gauges: str,
    sim_variable: str,
    obs_variable: str,
    base_path: pathlib.Path,
    breakpoints_path: pathlib.Path,
    metrics_to_calculate: Optional[list[str]] = METRICS,
    lead_times: list[str] = data_paths.LEAD_TIMES,
    time_periods: Optional[dict[str, list[str]]] = None,
    path_modifier: Optional[str] = None,
) -> dict[int, pd.DataFrame]:
    """Calculates metrics for many gauges."""

    # If a list of metrics is not provided, calculate all metrics.
    if metrics_to_calculate is None:
        metrics_to_calculate = metrics.get_available_metrics()

    # If time periods is not defined, make it not defined for all gauges.
    if time_periods is None:
        time_periods = {gauge: None for gauge in gauges}

    # Initialize storage. This should be a separate Pandas dataframe
    # for each metric, with index of gauges and columns of lead times.
    gauges_metrics = {
        metric: pd.DataFrame(index=gauges, columns=data_paths.LEAD_TIMES)
        for metric in metrics_to_calculate
    }

    for gauge in tqdm.tqdm(gauges):
        if gauge in ds.gauge_id.values:
            gauge_metrics_df = _calculate_and_save_metrics_for_one_gague(
                ds=ds,
                gauge=gauge,
                sim_variable=sim_variable,
                obs_variable=obs_variable,
                metrics_to_calculate=metrics_to_calculate,
                lead_times=lead_times,
                time_period=time_periods[gauge],
                base_path=breakpoints_path,
                path_modifier=path_modifier,
            )
    
        # Store the results in a dataframe formatted as above.
        for metric in metrics_to_calculate:
            gauges_metrics[metric].loc[gauge] = gauge_metrics_df.loc[metric]


    # Save metrics-specific files.
    for metric in metrics_to_calculate:
        _ = save_metrics_df(
            df=gauges_metrics[metric],
            metric=metric,
            base_path=base_path,
            path_modifier=path_modifier,
        )

    return gauges_metrics


def calculate_and_save_metrics_for_many_gagues_and_many_models(
    restart: bool,
    experiments: list[str],
    ds: dict[str, xarray.Dataset],
    gauges: str,
    sim_variable: str,
    obs_variable: str,
    base_path: pathlib.Path,
    breakpoints_path: pathlib.Path,
    metrics_to_calculate: Optional[list[str]] = METRICS,
    lead_times: list[str] = data_paths.LEAD_TIMES,
    time_periods: Optional[dict[str, list[str]]] = None,
) -> dict[int, pd.DataFrame]:
    """Calculates metrics for many gauges and many models."""

    if os.path.exists(base_path) and restart:
        shutil.rmtree(base_path)
    loading_utils.create_remote_folder_if_necessary(base_path)
    if os.path.exists(breakpoints_path) and restart:
        shutil.rmtree(breakpoints_path)
    loading_utils.create_remote_folder_if_necessary(breakpoints_path)

    gauge_metrics = {}
    for experiment in experiments:
        print(f'Working on experiment: {experiment}.')
        gauge_metrics[experiment] = calculate_and_save_metrics_for_many_gagues(
            ds=ds[experiment],
            gauges=gauges,
            sim_variable=sim_variable,
            obs_variable=obs_variable,
            base_path=base_path,
            breakpoints_path=breakpoints_path,
            metrics_to_calculate=metrics_to_calculate,
            path_modifier=experiment,
            time_periods=time_periods,
            lead_times=lead_times
        )

    return gauge_metrics

