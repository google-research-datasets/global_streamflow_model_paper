"""Utilities to calculate return period metrics."""

import os
import pathlib
import shutil
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm
import xarray

from backend import data_paths
from backend import evaluation_utils
from backend import loading_utils
from backend import metrics_utils
from backend.return_period_calculator import exceptions
from backend.return_period_calculator import return_period_calculator



RETURN_PERIOD_TIME_WINDOWS = [
    pd.Timedelta(0, unit='d'),
    pd.Timedelta(1, unit='d'),
    pd.Timedelta(2, unit='d'),
]


def _true_positives_fraction_in_window(
    a_crossovers: np.ndarray,
    b_crossovers: np.ndarray,
    discard_nans_in_window: bool,
    window_in_timesteps: int,
):
    """Calculates fraction of crossovers that were hit within a window.

    Handles NaNs by ignoring crossovers where array b is NaN anywhere
    within the window around crossovers in array a.

    Args:
    a_crossovers: np.ndarray
      First 0/1/NaN indicator array of crossovers.
    b_crossovers: np.ndarray
      Second 0/1/NaN indicator array of crossovers.
    discard_nans_in_window: bool
      True if you want to throw out all samples from 'a' where 'b' has any nans in the window.
      This is useful when 'b' are observations and you don't want to penalize a model due to
      the fact that the observed record is incomplete.
    window_in_timesteps: int
      Window around a crossover in a to search for a crossover in b.

    Returns:
    Fraction of crossovers in a where (1) the corresponding window in b contains
    no NaNs and (2) a corresponding crossover exists in b.
    """
    a_crossover_idxs = np.where((a_crossovers != 0) & ~np.isnan(a_crossovers))[0]
    b_crossover_idxs = np.where((b_crossovers != 0) & ~np.isnan(b_crossovers))[0]
    b_nan_crossover_idxs = np.where(np.isnan(b_crossovers))[0]

    # Count fraction of crossovers we hit.
    total_count = 0
    true_positives = 0
    for a_idx in a_crossover_idxs:
        # Do not count crossovers if there are NaNs in b within the window.
        if discard_nans_in_window and np.any(
            np.abs(b_nan_crossover_idxs - a_idx) <= window_in_timesteps + 1e-6
        ):
            continue
        total_count += 1
        if np.any(np.abs(b_crossover_idxs - a_idx) <= window_in_timesteps + 1e-6):
            true_positives += 1

#     import pdb; pdb.set_trace()

    if total_count > 0:
        return true_positives / total_count
    
    # This is a decision to define:
    #  -- Precision as 0 if there are no predicted events but there are obsreved events.
    #  -- Recall as 0 if there are no observed events but there are predicted events.
    elif len(a_crossover_idxs) == 0 and len(b_crossover_idxs) > 0:
        return 0

    # This is a decision to define both precision and recall as 1 if there are no events to
    # predict *and* the model predicts no events.
    elif len(a_crossover_idxs) == 0 and len(b_crossover_idxs) == 0:
        return 1
    
    # This is a decision to define:
    #  -- Precision as 1 there are no observed data around any perdicted event.
    #  -- Recall as 1 if there are no predicted data around any observed event.
    # The second option should !not! be used. When the 'a' series are observations,
    # (i.e., calculating recall), you should set `discard_nans_in_window` to False.
    elif len(a_crossover_idxs) > 0 and discard_nans_in_window:
        return 1
        
    else:
        print(total_count, true_positives, len(a_crossover_idxs), len(b_crossover_idxs))
        print('You should only get here if there are nans in your predictions. '
              'If you see this message, please debug.')
        return np.nan


def _single_return_period_performance_metric(
    observations: np.ndarray,
    simulations: np.ndarray,
    obs_return_period_calculator: return_period_calculator
    .ReturnPeriodCalculator,
    sim_return_period_calculator: return_period_calculator
    .ReturnPeriodCalculator,
    return_period: float,
    window_in_timesteps: int,
) -> Tuple[Mapping[str, float], float]:
    """Calculates hit/miss rates for a single return period and time window."""

    # Calculate flow values at return period. These can be different for obs and
    # sim, and we want to test based on the model climatology.
    if sim_return_period_calculator:
        sim_flow_value = sim_return_period_calculator.flow_value_from_return_period(
            return_period)
    else:
        sim_flow_value = np.nan
    if obs_return_period_calculator:
        obs_flow_value = obs_return_period_calculator.flow_value_from_return_period(
            return_period)
    else:
        return {'precision': np.nan, 'recall': np.nan}, sim_flow_value

    # Find obs points crossing return period threshold.
    above_threshold = np.where(
        np.isnan(observations),
        np.nan,
        (observations >= obs_flow_value).astype(float),
    )
    obs_crossovers = np.maximum(0, np.diff(above_threshold))

    # Find sim points crossing return period threshold.
    above_threshold = np.where(
        np.isnan(simulations),
        np.nan,
        (simulations >= sim_flow_value).astype(float),
    )
    sim_crossovers = np.maximum(0, np.diff(above_threshold))

#     import pdb; pdb.set_trace()
    precision = _true_positives_fraction_in_window(
        sim_crossovers, obs_crossovers, True, window_in_timesteps
    )
    
    recall = _true_positives_fraction_in_window(
        obs_crossovers, sim_crossovers, False, window_in_timesteps
    )

    return {'precision': precision, 'recall': recall}, sim_flow_value


def calculate_return_period_performance_metrics(
    observations: pd.Series,
    predictions: pd.Series,
    temporal_resolution: str = '1D',
    evaluation_time_period: Optional[list[str]] = None,
) -> Tuple[Mapping[str, float], Mapping[str, float]]:
  """Calculates metrics for cross-product of return periods and time windows."""

  # Convert temporal resolution to timedelta.
  temporal_resolution = pd.Timedelta(temporal_resolution)

  # Convert unix timestamp index into datetime index.
  predictions.index = [
      pd.to_datetime(t, unit='s', origin='unix') for t in predictions.index]
  observations.index = [
      pd.to_datetime(t, unit='s', origin='unix') for t in observations.index]

  # Fit return period distributions.
  try:
    obs_return_period_calculator = return_period_calculator.ReturnPeriodCalculator(
        hydrograph_series=observations,
        hydrograph_series_frequency=temporal_resolution,
        use_simple_fitting=True,
        verbose=False,
    )
  except exceptions.NotEnoughDataError:
    obs_return_period_calculator = None

  try:
    sim_return_period_calculator = return_period_calculator.ReturnPeriodCalculator(
        hydrograph_series=predictions,
        hydrograph_series_frequency=temporal_resolution,
        use_simple_fitting=True,
        verbose=False,
    )
  except exceptions.NotEnoughDataError:
    sim_return_period_calculator = None

  # Fill any missing dates in the datetime index. This is necessary so that
  # the window of separation is in constant units.
  start_date = min([min(observations.index), min(predictions.index)])
  end_date = max([max(observations.index), max(predictions.index)])
  new_date_range = pd.date_range(
      start=start_date,
      end=end_date,
      freq=temporal_resolution,
  )

  # Remove any duplicate timestamps and reindex so that the return period
  # calculator has a full timeseries to work with.
  observations = observations[~observations.index.duplicated(keep='first')]
  predictions = predictions[~predictions.index.duplicated(keep='first')]
  observations = observations.reindex(new_date_range, fill_value=np.nan)
  predictions = predictions.reindex(new_date_range, fill_value=np.nan)

  # Cut out the evaluation time period.
  if evaluation_time_period is not None:
    obs = observations.loc[slice(*evaluation_time_period)]
    sim = predictions.loc[slice(*evaluation_time_period)]
  else:
    obs = observations.copy()
    sim = predictions.copy()

  # If the calculators were fit, calculate performance metrics.
  return_period_metrics = {}
  sim_return_period_dict = {}
  for return_period in evaluation_utils.RETURN_PERIODS:
    for window in RETURN_PERIOD_TIME_WINDOWS:

      # Convert time window to an index in the correct time units.
      window_in_timesteps = window / temporal_resolution

      metrics, sim_return_period_dict[
          return_period] = _single_return_period_performance_metric(
              observations=obs.values,
              simulations=sim.values,
              obs_return_period_calculator=obs_return_period_calculator,
              sim_return_period_calculator=sim_return_period_calculator,
              return_period=return_period,
              window_in_timesteps=window_in_timesteps,
          )
    
      metric_name_base = f'return_period_{return_period}_window_{window_in_timesteps}_'
      return_period_metrics.update({
          metric_name_base + metric_name: metric_value
          for metric_name, metric_value in metrics.items()
      })

  # Save the simulation return periods by themselves.
  flow_values_from_return_periods = {
      f'return_period_{return_period}': flow_value
      for return_period, flow_value in sim_return_period_dict.items()
  }

  return return_period_metrics, flow_values_from_return_periods


def calculate_and_save_metrics_for_one_gague(
    ds: xarray.Dataset,
    gauge: str,
    sim_variable: str,
    obs_variable: str,
    base_path: pathlib.Path,
    temporal_resolution: str = '1D',
    lead_times: list[str] = data_paths.LEAD_TIMES,
    distribution_time_period: Optional[list[str]] = None,
    evaluation_time_period: Optional[list[str]] = None,
    path_modifier: Optional[str] = None,
) -> pd.DataFrame:
    """Calculates return period metrics for many gauges."""

    if gauge not in ds.gauge_id:
        return

    # Check if gauge file already exists.
    precision_filepath = metrics_utils.save_metrics_df(
        df=None,
        metric=gauge,
        base_path=base_path,
        path_modifier=f'{path_modifier}/precision',
    )
    recall_filepath = metrics_utils.save_metrics_df(
        df=None,
        metric=gauge,
        base_path=base_path,
        path_modifier=f'{path_modifier}/recall',
    )

    # Initialize storage.
    return_period_time_windows = [w/temporal_resolution for w in RETURN_PERIOD_TIME_WINDOWS]

    precision_metrics = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [evaluation_utils.RETURN_PERIODS, return_period_time_windows]),
        columns=data_paths.LEAD_TIMES
    )
    recall_metrics = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [evaluation_utils.RETURN_PERIODS, return_period_time_windows]),
        columns=data_paths.LEAD_TIMES
    )

    for lead_time in lead_times:
        if 'lead_time' in ds.dims:
            sim = ds.sel(gauge_id=gauge, lead_time=lead_time)[sim_variable].to_series()
            obs = ds.sel(gauge_id=gauge, lead_time=lead_time)[obs_variable].to_series()
        else:
            sim = ds.sel(gauge_id=gauge)[sim_variable].to_series()
            obs = ds.sel(gauge_id=gauge)[obs_variable].to_series()

        # Pull time period.
        if distribution_time_period is not None:
            obs = obs.sel(time=slice(*distribution_time_period))
            sim = sim.sel(time=slice(*distribution_time_period))

        # Calculate the metrics.
        metrics_dict, _ = calculate_return_period_performance_metrics(
          predictions=sim,
          observations=obs,
          evaluation_time_period=evaluation_time_period,
        )
    
        # Store calculated metrics in dataframe.
        for rp in evaluation_utils.RETURN_PERIODS:
            for w in return_period_time_windows:
                metric_name = f'return_period_{rp}_window_{w}_precision'
                precision_metrics.loc[(rp, w), lead_time] = metrics_dict[metric_name]
                metric_name = f'return_period_{rp}_window_{w}_recall'
                recall_metrics.loc[(rp, w), lead_time] = metrics_dict[metric_name]

    # Save metrics for this gauge.
    _ = metrics_utils.save_metrics_df(
        df=recall_metrics,
        metric=gauge,
        base_path=base_path,
        path_modifier=f'{path_modifier}/recall',
    )
    _ = metrics_utils.save_metrics_df(
        df=precision_metrics,
        metric=gauge,
        base_path=base_path,
        path_modifier=f'{path_modifier}/precision',
    )

    return

# --- Main function to call from notebooks ----------------------------------


def compute_metrics(
    restart: bool,
    working_path: pathlib.Path,
    experiments: list[str],
    gauge_list: list[str],
    ds_dict: dict[str, xarray.Dataset],
    sim_variable: str,
    obs_variable: str,
    evaluation_time_periods: Optional[list] = None,
    lead_times: Optional[list[int]] = None
) -> list[str]:
    """Call this function to compute return period metrics for an experiment."""
    
    if lead_times is None:
        lead_times = data_paths.LEAD_TIMES
    
    if os.path.exists(working_path) and restart:
        shutil.rmtree(working_path)
    loading_utils.create_remote_folder_if_necessary(working_path)
   
    if evaluation_time_periods is None:
        evaluation_time_periods = {gauge: None for gauge in gauge_list}

    missing_gauges = {experiment: [] for experiment in experiments}
    for experiment in experiments:
  
        print(f'Working on experiment: {experiment} ...')

        for gauge in tqdm.tqdm(gauge_list):
#             try:
                calculate_and_save_metrics_for_one_gague(
                    ds=ds_dict[experiment],
                    gauge=gauge,
                    sim_variable=sim_variable,
                    obs_variable=obs_variable,
                    base_path=working_path,
                    path_modifier=experiment,
                    evaluation_time_period=evaluation_time_periods[gauge],
                    lead_times=lead_times
      )
#             except:
#                 missing_gauges[experiment].append(gauge)

    return missing_gauges
                

