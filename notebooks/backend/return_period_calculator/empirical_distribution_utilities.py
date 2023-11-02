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

"""Code to estimate empirical distributions over peak streamflow values.

This code implements flood frequency interval calculations from guidelines
in the USGS Bulletin 17c (2019):

https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""
import numpy as np
from sklearn import linear_model

from backend.return_period_calculator import base_fitter

# Minimum number of bins allowed in histograms.
MIN_ALLOWED_HISTOGRAM_BINS = 20


def empirical_pdf(
    data: np.ndarray,
    min_allowed_bins: int = MIN_ALLOWED_HISTOGRAM_BINS,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
  """Estimates a probability mass function from empirical data.

  Args:
    data: Series of data from which to derive an empirical distribution.
    min_allowed_bins: Manual override for number of frequency bins in
        cases where the Freedman Diaconis method returns too few.

  Returns:
    - Probabiliy mass estimates integrated over bins: dims = (nbins, )..
    - Locations of bin centers: dims = (nbins, ).
    - Bin width -- all bins are the same width.
    - Bin edges: dims = (nbins+1, ).
  """
  # Freedman-Diaconis method for selecting the number of bins.
  inter_quartile_range = np.percentile(data, 75) - np.percentile(data, 25)
  freedman_diaconis_bin_width = 2*inter_quartile_range*len(data)**(-1/3)

  # Add one extra bin so that we can center on the min and max data.
  num_bins = int(
      (np.max(data) - np.min(data)) / freedman_diaconis_bin_width) + 1

  # Don't allow less than a specified number of bins.
  num_bins = np.maximum(min_allowed_bins, num_bins)

  # Create equally-spaced bins.
  bin_width = (np.max(data) - np.min(data)) / (num_bins - 1)
  bins = np.linspace(
      np.min(data) - bin_width / 2,
      np.max(data) + bin_width / 2, num_bins + 1)

  # Create the histogram. This returns a tuple of (frequency, bin edges).
  hist = np.histogram(data, bins=bins, density=True)

  # Bin centers from bin edges.
  bin_edges = hist[1]
  bin_centers = (bin_edges[1:] - bin_edges[:-1]) / 2 + bin_edges[:-1]

  # Probability mass from histogram.
  probability_mass = hist[0]

  return probability_mass, bin_centers, bin_width, bin_edges


def simple_empirical_plotting_position(
    data: np.ndarray,
    a: float = 0.,
) -> np.ndarray:
  """Empirical frequency distribution.

  This implements the plotting position formula in the main text of
  Bulletin 17c (Equation 2, page 23). This is superseded by the plotting
  position formula in Appendix 5, which is implemented in
  `_threshold_exceedance_plotting_position()`.

  Values of the `a` parameter come from Table 5-1 in Appendix 5:
    Weibull:    0 (Unbiased exceedance probabilities for all distributions)
    Cunnane:    0.40 (Approximately quantile-unbiased)
    Gringorten: 0.44 (Optimized for Gumbel distribution)
    Hazen:      0.50 (A traditional choice)

  Args:
    data: Series of data from which to derive an empirical distribution.
    a: Plotting position parameter (see Bulletin, page 23).

  Returns:
    Array of distribution-free plotting positions.

  Raises:
    ValueError if the data array is unsorted.
  """
  # This function expects to receive sorted data.
  if not np.all(np.diff(data) >= 0):
    raise ValueError('Data must be sorted in ascending order.')

  # `n` in Equation 2.
  record_length = len(data)

  # `i` in Equation 2.
  sorted_indexes = data.argsort()[::-1].argsort() + 1

  # `pi` in Equation 2.
  plotting_positions = (sorted_indexes - a) / (record_length + 1 - 2*a)

  return plotting_positions


def threshold_exceedance_empirical_plotting_position() -> np.ndarray:
  """Threshold-exceedance plotting position from Appendix 5."""
  # TODO(gsnearing): Implement threshold_exceedance_empirical_plotting_position.
  raise NotImplementedError()


class LogLogTrendFitter(base_fitter.BaseFitter):
  """Estimates return periods without any distribution using linear regression.

  This fits a regression from flow values to exceedance probabilities. It can be
  used as a last resort for estimating return periods from tiny samples.
  It is not reliable. Use with caution.
  """

  def __init__(
      self,
      data: np.ndarray,
      log_transform: bool = True
  ):
    """Constructor for log-trend fitting utility.

    Fits a linear trend line between log-transformed streamflow and
    log-transformed simple empirical plotting positions.

    Args:
      data: Flow peaks to fit in physical units.
      log_transform: Whether to transform the data before fitting a
        distribution.
    """
    super().__init__(
        data=data,
        log_transform=log_transform
    )

    # Regress from log-transformed streamflow onto log-transformed exceedance
    # probability.
    y_data = self._transform_data(
        simple_empirical_plotting_position(
            data=np.sort(self.transformed_sample))
    )
    x_data = np.expand_dims(self.transformed_sample, axis=1)
    self._regression = linear_model.LinearRegression().fit(x_data, y_data)

  @property
  def type_name(self) -> str:
    return self.__class__.__name__

  def _invert_fit_regression(self, y_values: np.ndarray) -> np.ndarray:
    # TODO(gsnearing): This can extrapolate beyond (0, 1), which causes
    # problems translating to RPs. We need a regression directly on the CDF.
    return (y_values - self._regression.intercept_) / self._regression.coef_

  def exceedance_probabilities_from_flow_values(
      self,
      flows: np.ndarray
  ) -> np.ndarray:
    """Predicts exceedance probabilities from streamflow values.

    Args:
      flows: Streamflow values in physical units.

    Returns:
      Predicted exceedance probabilities.

    Raises:
      ValueError if return periods are requested for zero-flows.
    """
    if np.any(flows <= 0):
      raise ValueError('All flow values must be positive.')
    transformed_flows = self._transform_data(flows)
    transformed_exceedance_probabilities = self._regression.predict(
        np.expand_dims(transformed_flows, axis=1))
    exceedance_probabilities = self._untransform_data(
        transformed_exceedance_probabilities)
    exceedance_probabilities[
        (exceedance_probabilities > 1) | (exceedance_probabilities < 0)
    ] = np.nan
    return exceedance_probabilities

  def flow_values_from_exceedance_probabilities(
      self,
      exceedance_probabilities: np.ndarray,
  ) -> np.ndarray:
    """Predicts from pre-fit log-linear regression.

    Args:
      exceedance_probabilities: Probability of exceeding a particular flow value
        in a given year.

    Returns:
      Flow values corresponding to requeseted exceedance_probabilities.

    Raises:
      ValueError if cumulative probailities are outside realistic ranges, or
        include 0 or 1.
    """
    base_fitter.test_for_out_of_bounds_probabilities(
        probabilities=exceedance_probabilities)
    transformed_exceedance_probabilities = self._transform_data(
        data=exceedance_probabilities)
    transformed_flow_values = self._invert_fit_regression(
        y_values=transformed_exceedance_probabilities)
    return self._untransform_data(data=transformed_flow_values)

