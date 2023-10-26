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

"""Functions are used to fit naive theoretical return period distributions.

These are the naive fitting methods described in the Bulletin, not the full
EMA procedure.

This code implements flood frequency interval calculations from guidelines
in the USGS Bulletin 17c (2019):

https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""
from typing import Tuple

import numpy as np
from scipy import special as scipy_special

from backend.return_period_calculator import base_fitter
from backend.return_period_calculator import exceptions

_MIN_DATA_POINTS = 7
_MIN_ALLOWED_SKEW = 0.000016  # This value is taken from the scipy Pearson3.


def pearson3_invcdf(
    alpha: float,
    beta: float,
    tau: float,
    quantiles: np.ndarray,
) -> np.ndarray:
  """Returns flow value for distribution quantiles.

  This is the inverse of `pearson3_cdf()`.

  Args:
    alpha: Log-Pearson-III distribution shape parameter.
    beta: Log-Pearson-III distribution scale parameter.
    tau: Log-Pearson-III distribution location parameter.
    quantiles: Distribution quantiles for which we want to know
      associated random variable values.

  Raises:
    ValueError if cumulative probailities are outside realistic ranges, or
      include 0 or 1.
  """
  # Ensure that the requested probabilities are realistic -- i.e., in (0, 1).
  base_fitter.test_for_out_of_bounds_probabilities(
      probabilities=quantiles)

  # Equation 11 in Bulletin 17c.
  if beta >= 0:
    inverse_gamma = scipy_special.gammaincinv(alpha, quantiles)
  else:
    inverse_gamma = scipy_special.gammainccinv(alpha, quantiles)
  return tau + beta*inverse_gamma


def pearson3_cdf(
    alpha: float,
    beta: float,
    tau: float,
    values: np.ndarray,
) -> np.ndarray:
  """Returns distribution quantiles for given flow values.

  This is the inverse of `pearson3_invcdf()`.

  See Bulletin 17c pg. 25 for description of the notation used for distribution
  parameters.

  Args:
    alpha: Log-Pearson-III distribution shape parameter.
    beta: Log-Pearson-III distribution scale parameter.
    tau: Log-Pearson-III distribution location parameter.
    values: Random variable values for which we want to know distribution
      quantiles.
  """
  # This is the inverse of Equation 11 in Bulletin 17c.
  # The logic based on beta comes from the following reference:
  #   Tegos, Sotiris A., et al. "New results for Pearson type III family of
  #   distributions and application in wireless power transfer." IEEE Internet
  #   of Things Journal (2022).
  # Incidentally, this depnedence on beta is actaully a dependence on skew,
  # since beta and skew have the same sign. Old versions of the scipy
  # implementation of the Pearson III distribution forgot this, which caused
  # an error for negative skew.
  if beta >= 0:
    return scipy_special.gammainc(alpha, (values - tau) / beta)
  else:
    return scipy_special.gammaincc(alpha, (values - tau) / beta)


def pearson3_pmf(
    alpha: float,
    beta: float,
    tau: float,
    edges: np.ndarray,
) -> np.ndarray:
  """Returns probability mass between two flow values.

  See Bulletin 17c pg. 25 for description of the notation used for distribution
  parameters.

  Args:
    alpha: Log-Pearson-III distribution shape parameter.
    beta: Log-Pearson-III distribution scale parameter.
    tau: Log-Pearson-III distribution location parameter.
    edges: Sequence of values that define edges of the mass function.
  """
  # Calculate the CDFs at each flow value.
  cumulative_probabilities = pearson3_cdf(alpha, beta, tau, edges)

  # Take probability mass as the difference between CDFs.
  return np.diff(cumulative_probabilities)


def sample_moments(
    data: np.ndarray,
) -> Tuple[float, float, float]:
  """Estimate sample moments for log-Pearson III distribution.

  These moments are identical to equations (5) through (7). The formulas
  used in this function come from equations (7-11) and (7-14).

  Args:
    data: log-transformed data from which to estimate moments.

  Returns:
    First three central moment estimates.

  Raises:
    NotEnoughDataError if not enough data.
  """
  n = len(data)
  if n < _MIN_DATA_POINTS:
    raise exceptions.NotEnoughDataError(
        num_data_points=n,
        data_requirement=_MIN_DATA_POINTS,
        routine='Sample Pearson II Moments (tdu.sample_moments)',
    )
  mean = np.mean(data)
  std = n / (n-1) * np.sqrt(np.mean((data - mean)**2))
  skew = np.sqrt(n * (n-1)) / (n-2) * np.mean(((data - mean)**3) / (std**3))
  return mean, std, skew


def parameters_from_moments(
    mean: float,
    std: float,
    skew: float,
) -> Tuple[float, float, float]:
  """Log-Pearson III distribution parameters from sample moments.

  These are equations (7-13) in Bulletin 17c. Notably, these are *not* equations
  (8) through (10), which are not the same.

  Args:
    mean: Mean (mu) of the distribution.
    std: Standard deviation (sigma) of the distribution.
    skew: Skew (gamma) of the distribution.

  Returns:
    Parameters of the distribution: alpha (shape), beta (scale), tau (location).

  Raises:
    NumericalFittingError if skew is too small, which can cause numerical
    errors.
  """
  if abs(skew) < _MIN_ALLOWED_SKEW:
    raise exceptions.NumericalFittingError(
        routine='Estimating Pearson III parameters',
        condition='Small Skew',
    )
  alpha = 4 / skew**2
  beta = std * skew / 2
  tau = mean - 2 * std / skew
  return alpha, beta, tau


def pearson3_parameters_simple(
    data: np.ndarray,
) -> dict[str, float]:
  """Implements parameter estimation for the Simple Case.

  The 'simple case' means that there are no potentially influential low flood
  (PILF) values, and also no historical record. The strategy in this case is
  to estimate Log-Pearson III parameters using sample moments. Distribution
  parameters are named as in the Bulletin 17c (alpha, beta, tau).

  This fitting procedure is described on page 25 of Bulletin 17c.

  Args:
    data: Array of annual maximum flow values that are to be used to
      fit the return period distribution. Input in transformed (not physical)
      units.

  Returns:
    Mapping of fit parameters keyed by parameter name according to
    equations 8-10: alpha (shape), beta (scale), tau (location).
  """
  mean, std, skew = sample_moments(data)
  alpha, beta, tau = parameters_from_moments(mean, std, skew)
  return {'alpha': alpha, 'beta': beta, 'tau': tau}


class SimpleLogPearson3Fitter(base_fitter.BaseFitter):
  """Estimates return periods using standard MLE distribution fitting.

  This cannot handle outliers, thresholds, regional skew, etc.
  """

  def __init__(
      self,
      data: np.ndarray,
      log_transform: bool = True
  ):
    """Constructor for a log-Pearson III distribution fitter.

    Fits parameters of a log-Pearson-III distribution from data.

    Args:
      data: Flow peaks to fit in physical units.
      log_transform: Whether to transform the data before fitting a
        distribution.
    """
    super().__init__(
        data=data,
        log_transform=log_transform
    )
    self._distribution_parameters = pearson3_parameters_simple(
        data=self.transformed_sample)
    self._sample_moments = sample_moments(self.transformed_sample)

  @property
  def type_name(self) -> str:
    return self.__class__.__name__

  def exceedance_probabilities_from_flow_values(
      self,
      flows: np.ndarray,
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
    return 1 - pearson3_cdf(
        alpha=self._distribution_parameters['alpha'],
        beta=self._distribution_parameters['beta'],
        tau=self._distribution_parameters['tau'],
        values=transformed_flows,
    )

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
    transformed_flow_values = pearson3_invcdf(
        alpha=self._distribution_parameters['alpha'],
        beta=self._distribution_parameters['beta'],
        tau=self._distribution_parameters['tau'],
        quantiles=(1 - exceedance_probabilities),
    )
    return self._untransform_data(data=transformed_flow_values)
