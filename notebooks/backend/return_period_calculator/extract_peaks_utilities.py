# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions to extract peaks from a flow series.

These peak values are what will be used to fit a return period distribution.
The correct way to do this is to extract annual maximum flow values, which
is done using the `extract_annual_maximums()` function. However, there are
some situations where other types of filtering is useful, so alternative
routines are provided for such cases. We strongly recommend only using
`extract_annual_maximums()'

This code implements flood frequency interval calculations from guidelines
in the USGS Bulletin 17c (2019):

https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

from backend.return_period_calculator import exceptions

_DEFAULT_TIME_WINDOW_FOR_PEAKS = pd.Timedelta(30, unit='d')
_DEFAULT_PERCENTILE_FOR_PEAKS = 95
_MIN_PEAK_PERCENTILE = 70
_MIN_FRACTION_OF_WATER_YEAR_IN_RECORD = 1/2


def extract_annual_maximums(
    hydrograph_series: pd.Series,
    frequency: pd.Timedelta,
    min_fraction_of_records_per_year: Optional[float] = None,
) -> pd.Series:
  """Extracts the annual maximum values from a discharge timeseries.

  Return period calculations are done on annual maximums.

  Args:
    hydrograph_series: Pandas series containing the flow record indexed by
      timestamp.
    frequency: Time frequency of the index in the input series.
    min_fraction_of_records_per_year: Fraction of records at the frequency
      of the data needed to extract an annual peak.

  Returns:
    DataFrame containing annual maximum values indexed by the full timestamp
    from which those values were recorded in the original df.
  """
  if min_fraction_of_records_per_year is None:
    min_fraction_of_records_per_year = _MIN_FRACTION_OF_WATER_YEAR_IN_RECORD

  # Determine the water year.
  water_year = hydrograph_series.index.year.where(
      hydrograph_series.index.month < 10, hydrograph_series.index.year + 1)

  # Number of records per year based on time frequency of the dataset.
  records_per_year = pd.Timedelta(365, unit='d') / frequency

  annual_max_value, annual_max_date = [], []
  for _, grp in hydrograph_series.groupby(water_year):
    if (len(grp.dropna()) <
        min_fraction_of_records_per_year * records_per_year):
      continue
    annual_max_value.append(grp.max())
    annual_max_date.append(grp.idxmax())
  return pd.Series(annual_max_value, index=annual_max_date).dropna()


def extract_peaks_by_separation_and_threshold(
    hydrograph_series: pd.Series,
    frequency: pd.Timedelta,
    window: Optional[pd.Timedelta],
    percentile: Optional[float],
) -> pd.Series:
  """Extract peaks from a dicharge timeseries defined by separation and value.

  **Do not use this unless you fully understand the statistical implications.**

  This is an alternative (incorrect) way of extracting a peak flow timeseries
  for calculating return periods. This works by finding peaks with a given
  degree of separation and a given magnitude.

  Args:
    hydrograph_series: Pandas series containing the flow record indexed by
      timestamp.
    frequency: Time frequency of the index in the input series.
    window: Time window of minimum separation between peaks.
    percentile: Threshold for defining a peak expressed in percentile of the
      flow record in (0, 100).

  Returns:
    Series containing flow peak values indexed by the full timestamp
    from which those values were recorded in the original df.

  Raises:
    NotEnoughDataError if the hydrograph is empty.
  """
  # Check for empty hydrographs.
  if hydrograph_series.empty:
    raise exceptions.NotEnoughDataError(
        num_data_points=0,
        data_requirement=1,
        routine='extract_peaks_by_separation_and_threshold',
    )
  if percentile is None:
    percentile = _DEFAULT_PERCENTILE_FOR_PEAKS

  # Fill any missing dates in the datetime index. This is necessary so that
  # the window of separation is in constant units.
  new_date_range = pd.date_range(
      start=hydrograph_series.index[0],
      end=hydrograph_series.index[-1],
      freq=frequency,
  )
  padded_hydrograph_series = hydrograph_series.reindex(
      new_date_range, fill_value=np.nan)

  # Convert window to a number of timesteps.
  if window is None:
    window = _DEFAULT_TIME_WINDOW_FOR_PEAKS
  window_in_timesteps = window / frequency

  # Compute magnitude threshold from percentile threshold.
  timeseries_array = np.squeeze(padded_hydrograph_series.values)
  threshold = np.nanpercentile(timeseries_array, q=percentile)
  timeseries_array_replace_nans = np.nan_to_num(
      timeseries_array, nan=-np.inf, copy=True)

  # Identify peaks and extract from the DataFrame.
  peaks_idx = scipy_signal.find_peaks(
      timeseries_array_replace_nans,
      distance=window_in_timesteps,
      height=threshold
  )[0]
  peaks = padded_hydrograph_series.iloc[peaks_idx]

  # Return extracted peaks.
  return peaks.dropna()


def extract_n_highest_peaks(
    hydrograph_series: pd.DataFrame,
    frequency: pd.Timedelta,
    window: Optional[pd.Timedelta],
    number_of_peaks: Optional[int] = None,
) -> pd.Series:
  """Extracts top N peaks from hydrograph with specified separation window.

  **Do not use this unless you fully understand the statistical implications.**

  This is an alternative (incorrect) way of extracting a peak flow timeseries
  for calculating return periods. This works by finding peaks with a given
  degree of separation and a given magnitude.

  Args:
    hydrograph_series: Pandas series containing the flow record indexed by
    frequency: Time frequency of the index in the input series.
    window: Time window of minimum separation between peaks.
    number_of_peaks: Number of highest peaks to return.

  Returns:
    Array containing flow peak values indexed by the full timestamp
    from which those values were recorded in the original df.

  Raises:
    ValueError if not enough peaks in the timeseries.
  """

  # Fill any missing dates in the datetime index. This is necessary so that
  # the window of separation is in constant units.
  new_date_range = pd.date_range(
      start=hydrograph_series.index[0],
      end=hydrograph_series.index[-1],
      freq=frequency,
  )
  padded_hydrograph_series = hydrograph_series.reindex(
      new_date_range, fill_value=np.nan)

  # The default number of peaks is the number of years in the
  # discharge hydrograph.
  if number_of_peaks is None:
    number_of_peaks = len(hydrograph_series.index.year.unique())

  # Convert window to a number of timesteps.
  if window is None:
    window = _DEFAULT_TIME_WINDOW_FOR_PEAKS
  window_in_timesteps = window / frequency

  # Find the magnitude threshold where we get enough peaks.
  # This is better than setting the threshold low because of the
  # time window.
  timeseries_array = np.squeeze(padded_hydrograph_series.values)
  timeseries_array_replace_nans = timeseries_array.copy()
  timeseries_array_replace_nans[
      np.isnan(timeseries_array_replace_nans)] = -np.inf
  for percentile in range(100, 0, -1):
    threshold = np.nanpercentile(timeseries_array, q=percentile)
    peaks_idx = scipy_signal.find_peaks(
        timeseries_array_replace_nans,
        distance=window_in_timesteps,
        height=threshold
    )[0]
    if len(peaks_idx) >= number_of_peaks:
      sorted_idx = np.argsort(
          padded_hydrograph_series.iloc[peaks_idx].values)[::-1]
      peaks = padded_hydrograph_series.iloc[
          peaks_idx[sorted_idx[:number_of_peaks]]]
      return peaks.dropna()
    elif percentile < _MIN_PEAK_PERCENTILE:
      # If we get here, it means that we did not find enough peaks even at the
      # lowest percentile that was checked.
      raise exceptions.NotEnoughDataError(
          data_requirement=number_of_peaks,
          num_data_points=len(peaks_idx),
          routine=f'extract_n_highest_peaks ({percentile}%)',
      )

