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

"""Primary object for calculating return periods.

All helper functions in other files in this codebase support this object.

This code implements flood frequency interval calculations from guidelines
in the USGS Bulletin 17c (2019):

https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""

import logging
from typing import Callable, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.return_period_calculator import empirical_distribution_utilities as edu
from backend.return_period_calculator import exceptions
from backend.return_period_calculator import extract_peaks_utilities
from backend.return_period_calculator import generalized_expected_moments_algorithm as gema
from backend.return_period_calculator import plotting_utilities
from backend.return_period_calculator import theoretical_distribution_utilities as tdu

_DEFAULT_PLOTTING_RETURN_PERIODS = np.array([1.01, 2, 5, 10, 20, 50, 100])
# The minimum years of record must be 2 or larger, since 2 are necessary to
# fit a linear trend.
_MIN_YEARS_OF_RECORD = 5


class ReturnPeriodCalculator():
  """Primary object for calculating return periods."""

  def __init__(
      self,
      peaks_series: Optional[pd.Series] = None,
      hydrograph_series: Optional[pd.Series] = None,
      hydrograph_series_frequency: Optional[pd.Timedelta] = None,
      is_stage: bool = False,
      extract_peaks_function: Callable[
          [pd.Series, pd.Timedelta],
          pd.Series
      ] = extract_peaks_utilities.extract_annual_maximums,
      use_simple_fitting: bool = False,
      use_log_trend_fitting: bool = False,
      kn_table: Optional[Mapping[int, float]] = None,
      verbose: bool = True,
  ):
    """Constructor for Return Period Calculator.

    Args:
      peaks_series: Option to allow users to supply their own peaks, instead
        of using utilities to extract peaks from a hydrograph.
      hydrograph_series: Systematic data record as a Pandas series in physical
        units (e.g., cms, m3/s) with dates as indexes. Peaks will be extracted
        from this hydrograph unless provided by `peaks_series`.
      hydrograph_series_frequency: Frequency of timestep in the hydrgraph
        series. Must be supplied if peaks_series is not supplied.
      is_stage: Indicates whether the hydrograph and/or peaks data are stage
        (as opposed to discharge). If stage data are used, they are not
        log-transformed.
      extract_peaks_function: Function to find "floods" to use for fitting a
        return period distribution. The default is annual maximums, and this
        should not be changed unless you know what you are doing and why.
      use_simple_fitting: Use simple distribution fitting instead of the
        Expected Moments Algorithm (EMA). This does not account for Potentially
        Impactful Low Floods (PILFs), zero-flows, or historical flood data.
      use_log_trend_fitting: Use log-linear regression on empirical plotting
        positions to fit a return period estimator, instead of fitting a
        distribution.
      kn_table: Custom test statistics table to override the Kn Table from
        Bulletin 17b in the Standard Grubbs Beck Test (GBT).
      verbose: Whether to print status messages during runtime.

    Raises:
      ValeError if neither hydrograph nor peaks are provided.
    """
    # Extract peaks or work with peaks supplied by the user.
    if hydrograph_series is not None and peaks_series is None:
      self._hydrograph = hydrograph_series
      if hydrograph_series_frequency is None:
        raise ValueError('User must supply the time frequency of the '
                         'hydrograph series.')
      self._peaks = extract_peaks_function(
          hydrograph_series,
          hydrograph_series_frequency,
      )
    elif hydrograph_series is not None and peaks_series is not None:
      self._hydrograph = hydrograph_series
      self._peaks = peaks_series.dropna()
    elif hydrograph_series is None and peaks_series is not None:
      self._hydrograph = peaks_series
      self._peaks = peaks_series.dropna()
    else:
      raise ValueError('Must supply either a hydrograph series or a peaks '
                       'series.')

    if len(self._peaks) < _MIN_YEARS_OF_RECORD:
      raise exceptions.NotEnoughDataError(
          num_data_points=len(self._peaks),
          data_requirement=_MIN_YEARS_OF_RECORD,
          routine='Return Period Calculator',
      )

    # If working with stage data, don't log-transform.
    self.is_stage = is_stage

    # TODO(gsnearing): Implement record extension with nearby sites (Appdx. 8).

    # Fit the distribution. First, we try using the Generalized Expected Moments
    # Algorithm, which is the standard approach in Bulletin 17c. If that fails
    # (e.g., sample size is too small), we revert to log-log linear regression
    # against simple empircal plotting positions. The user can request simple
    # distribution fitting instead of GEMA, and this also reverts to regression
    # if the sample size is too small or if there is any other type of numerical
    # error.
    run_backup_fitter = False
    try:
      if use_log_trend_fitting:
        raise exceptions.AskedForBackupError(method='Log-Liner Regression')

      if use_simple_fitting:
        self._fitter = tdu.SimpleLogPearson3Fitter(
            data=self._peaks.values,
            log_transform=(not self.is_stage)
        )
      else:
        self._fitter = gema.GEMAFitter(
            data=self._peaks.values,
            kn_table=kn_table,
            log_transform=(not self.is_stage)
        )
    except (exceptions.NumericalFittingError, exceptions.NotEnoughDataError):
      if verbose:
        logging.exception('Reverting to using the regression fitter as a '
                          'backup')
      run_backup_fitter = True
    except exceptions.AskedForBackupError:
      run_backup_fitter = True

    if run_backup_fitter:
      self._fitter = edu.LogLogTrendFitter(
          data=self._peaks.values,
          log_transform=(not self.is_stage)
          )

  def __len__(self) -> int:
    return len(self._peaks)

  @property
  def fitter_type(self) -> str:
    return self._fitter.type_name

  def plot_hydrograph_with_peaks(self):
    """Plot the hydrograph with values used for return period analysis."""
    plotting_utilities.plot_hydrograph_with_peaks(
        hydrograph_series=self._hydrograph,
        peaks_series=self._peaks,
    )
    plt.show()

  # TODO(gsnearing): Make this show zeros, PILFs, historical.
  def plot_fitted_distribution(self):
    """Plot the empirical and theoretical (fit) floods distributions."""
    plotting_utilities.plot_fitted_distribution(
        data=self._peaks,
        fitter=self._fitter,
    )
    plt.show()

  def plot_exceedence_probability_distribution(self):
    """Plot the exceedence probability distribution."""
    plotting_utilities.plot_exceedence_probability_distribution(
        fitter=self._fitter,
    )
    plt.show()

  def plot_hydrograph_with_return_periods(
      self,
      return_periods: Optional[np.ndarray] = None,
  ):
    """Plot hydrograph with overlaid return periods."""
    if return_periods is None:
      return_periods = _DEFAULT_PLOTTING_RETURN_PERIODS
    return_period_values = self.flow_values_from_return_periods(
        return_periods=return_periods)
    plotting_utilities.plot_hydrograph_with_return_periods(
        hydrograph_series=self._hydrograph,
        return_period_values={rp: val for rp, val in zip(
            return_periods, return_period_values)},
    )
    plt.show()

  def flow_values_from_return_periods(
      self,
      return_periods: np.ndarray,
  ) ->  np.ndarray:
  # TODO(gsnearing): Also return confidence intervals.
    """Flow values for an array of return periods.

    Args:
      return_periods: Return periods for which to calculate flow values.

    Returns:
      Estimated flow values in physical units for given return periods.
    """
    return self._fitter.flow_values_from_return_periods(
        return_periods=return_periods)

  def flow_value_from_return_period(
      self,
      return_period: float,
  ) ->  float:
    """Flow value for a single return period.

    Args:
      return_period: Return period for which to calculate flow value.

    Returns:
      Estimated flow value in physical units for a given return period.
    """
    return self._fitter.flow_values_from_return_periods(
        return_periods=np.array([return_period]))[0]

  def flow_values_from_percentiles(
      self,
      percentiles: np.ndarray,
  ) ->  np.ndarray:
  # TODO(gsnearing): Also return confidence intervals.
    """Flow values for an array of distribution percentiles.

    Args:
      percentiles: CDF percentiles for which to calculate flow values.

    Returns:
      Estimated flow values in physical units for given return periods.
    """
    return self._fitter.flow_values_from_exceedance_probabilities(
        exceedance_probabilities=1-percentiles)

  def percentiles_from_flow_values(
      self,
      flows: np.ndarray,
  ) ->  np.ndarray:
  # TODO(gsnearing): Also return confidence intervals.
    """CDF percentiles for a given set of flow values.

    Args:
      flows: flow values in physical units for given return periods

    Returns:
      Estimated CDF percentiles for which to calculate flow values.
    """
    return self._fitter.exceedance_probabilities_from_flow_values(
        flows=flows)

  def return_periods_from_flow_values(
      self,
      flows: np.ndarray,
  ) ->  np.ndarray:
  # TODO(gsnearing): Also return confidence intervals.
    """Return period for a given flow value.

    Args:
        flows: Flow values for which to calculate a return period in physical
          units.

    Returns:
      Estimated return period for given flow values.
    """
    mask = np.where(flows > 0)
    return_periods = np.full_like(flows, np.nan, dtype=float)
    return_periods[mask] = self._fitter.return_periods_from_flow_values(
        flows=flows[mask])
    return return_periods
