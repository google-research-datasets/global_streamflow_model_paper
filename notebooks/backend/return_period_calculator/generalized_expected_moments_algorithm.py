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

"""Implements the generalized EMA algorithm for fitting a log-Pearson III dist.

This code implements flood frequency interval calculations from guidelines
in the USGS Bulletin 17c (2019):

https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""
from typing import Mapping, Optional

import numpy as np
from scipy import special as scipy_special

from backend.return_period_calculator import base_fitter
from backend.return_period_calculator import exceptions
from backend.return_period_calculator import grubbs_beck_tester as gbt
from backend.return_period_calculator import theoretical_distribution_utilities as tdu

# This default value is defined on page 83.
_DEFAULT_CONVERGNCE_THRESHOLD = 1e-10
_MAX_ITERATIONS = 1000

# pylint: disable=g-complex-comprehension

# TODO(gsnearing): Add historical and gap data to EMA.
# TODO(gsnearing): Add regional skew to EMA.


def _parameters_from_central_moments(
    moments: dict[str, float],
) -> dict[str, float]:
  """Estimate distribution parameters from central moments."""
  # This is equation 7-13 (page 83).
  alpha, beta, tau = tdu.parameters_from_moments(
      mean=moments['M'],
      std=moments['S'],
      skew=moments['G'],
  )
  return {'alpha': alpha, 'beta': beta, 'tau': tau}


def _central_moments_from_data(
    data: np.ndarray,
) -> dict[str, float]:
  """Estimate distribution parameters from sample."""
  mean, std, skew = tdu.sample_moments(data=data)
  return {'M': mean, 'S': std, 'G': skew}


def _partial_gamma_integral(
    alpha: float,
    lower: float,
    upper: float,
) -> float:
  upper_gamma = scipy_special.gammainc(alpha, max(0, upper))
  lower_gamma = scipy_special.gammainc(alpha, max(0, lower))
  alpha_gamma = scipy_special.gamma(alpha)
  return alpha_gamma * (upper_gamma - lower_gamma)


def _expected_value_on_interval(
    lower_bound: float,
    upper_bound: float,
    alpha: float,
    beta: float,
    tau: float,
    moment: int,
) -> float:
  """Estimate central moment on interval data."""
  # Implements equation 7-18.

  if upper_bound == lower_bound:
    return upper_bound**moment

  scaled_lower = (lower_bound - tau) / beta
  scaled_upper = (upper_bound - tau) / beta
  if beta < 0:
    denominator = _partial_gamma_integral(
        alpha=alpha,
        lower=scaled_upper,
        upper=scaled_lower,
    )
  else:
    denominator = _partial_gamma_integral(
        alpha=alpha,
        lower=scaled_lower,
        upper=scaled_upper,
    )

  summation = 0
  for j in range(moment + 1):
    if beta < 0:
      numerator = _partial_gamma_integral(
          alpha=alpha+j,
          lower=scaled_upper,
          upper=scaled_lower,
      )
    else:
      numerator = _partial_gamma_integral(
          alpha=alpha+j,
          lower=scaled_lower,
          upper=scaled_upper,
      )
    summation += scipy_special.comb(moment, j) * beta**j * tau**(moment - j) * (
        numerator / denominator)

  return summation


def _expected_centralized_value_on_interval(
    lower_bound: float,
    upper_bound: float,
    alpha: float,
    beta: float,
    tau: float,
    mean: float,
    moment: int,
) -> float:
  """..."""
  summation = 0.
  for l in range(moment + 1):
    interval_expected_value = _expected_value_on_interval(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        alpha=alpha,
        beta=beta,
        tau=tau,
        moment=l,
    )
    summation += (
        scipy_special.comb(moment, l) *
        interval_expected_value *
        (-mean)**(moment - l)
    )
  return summation


def _central_moments_from_data_and_parameters(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    parameters: dict[str, float],
) -> dict[str, float]:
  """Estimate central moments from data and distribution parameters."""

  # Number of samples.
  num_samples = len(lower_bounds)

  # Estimate first moment of distribution (equation 7-15).
  interval_expected_values1 = [
      _expected_value_on_interval(
          lower_bound=lb,
          upper_bound=ub,
          alpha=parameters['alpha'],
          beta=parameters['beta'],
          tau=parameters['tau'],
          moment=1,
      ) for (lb, ub) in zip(lower_bounds, upper_bounds)
  ]
  mean = 1 / num_samples * np.sum(interval_expected_values1)

  # Estimate second moment of distribution (equation 7-15).
  interval_expected_values2 = [
      _expected_centralized_value_on_interval(
          lower_bound=lb,
          upper_bound=ub,
          alpha=parameters['alpha'],
          beta=parameters['beta'],
          tau=parameters['tau'],
          mean=mean,
          moment=2,
      ) for (lb, ub) in zip(lower_bounds, upper_bounds)
  ]
  stdev = np.sqrt(1 / (num_samples - 1) * np.sum(interval_expected_values2))

  # Estimate third moment of distribution (equation 7-15).
  interval_expected_values3 = [
      _expected_centralized_value_on_interval(
          lower_bound=lb,
          upper_bound=ub,
          alpha=parameters['alpha'],
          beta=parameters['beta'],
          tau=parameters['tau'],
          mean=mean,
          moment=3,
      ) for (lb, ub) in zip(lower_bounds, upper_bounds)
  ]
  skew = num_samples / stdev**3 / (num_samples - 1) / (
      num_samples - 2) * np.sum(interval_expected_values3)

  return {'M': mean, 'S': stdev, 'G': skew}


def _check_convergence_norm1(
    current_moments: dict[str, float],
    previous_moments: dict[str, float],
    convergence_threshold: Optional[float] = None,
) -> bool:
  """Checks whether the EMA algorithm has converged using first norm."""
  # Set default convergence threshold if a value is not supplied.
  if not convergence_threshold:
    convergence_threshold = _DEFAULT_CONVERGNCE_THRESHOLD

  # Calculate first norm.
  differences = [
      current_moments[key] - previous_moments[key] for key in current_moments
  ]
  norm_of_differences = np.sum(np.abs(differences))

  # Check norm against threshold.
  return norm_of_differences < convergence_threshold


def _ema(
    systematic_record: np.ndarray,
    pilf_threshold: float,
    convergence_threshold: Optional[float] = None,
) -> dict[str, float]:
  """Implements the Genearlized Expected Moments Algorithm (EMA).

  This is the full fitting procedure from Bulletin 17c, and is the main
  difference between that and the 1981 USGS protocol from Bulletin 17b.
  This algorithm is described on page 27 of Bulletin 17c, with full
  implementation details given in Appendix 7 (page 82).

  Args:
    systematic_record: Systematic data record of flood peaks.
      Must be in transformed units.
    pilf_threshold: As determined by the MGBT test. Units must match
      systematic record.
    convergence_threshold: Convergence threshold to be applied to the first
      norm of the moments of the EMA-estimated distribution. The default value
      is 1e-10.

  Returns:
    dict of fit parameters keyed by parameter name according to Equations 8-10.

  Raises:
    NumericalFittingError if there are nans or infs in iterative algorithm.
  """
  # Turn all data into lower and upper bounds.
  num_pilfs = len(systematic_record[systematic_record < pilf_threshold])
  lower_bounds = systematic_record[systematic_record >= pilf_threshold]
  upper_bounds = systematic_record[systematic_record >= pilf_threshold]
  if num_pilfs > 0:
    lower_bounds = np.concatenate([
        np.full(num_pilfs, -np.inf),
        lower_bounds,
    ])
    upper_bounds = np.concatenate([
        np.full(num_pilfs, pilf_threshold),
        upper_bounds,
    ])

  # Steps in this algorithm are listed on pages 83-84.
  # Step #1a: Initial estimates of central moments.
  # These are used for the first expected value calculations and also for the
  # first convergence check.
  previous_moments = _central_moments_from_data(data=systematic_record)

  # Step #2: Expectation-maximization loop.
  converged = False
  iteration = 0
  while not converged:
    iteration += 1

    # Update distribution parameters.
    parameters = _parameters_from_central_moments(moments=previous_moments)

    # Step #2a: Update expected moments.
    expected_moments = _central_moments_from_data_and_parameters(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        parameters=parameters,
    )

    # Error checking.
    if (np.isnan([val for val in expected_moments.values()]).any()
        or np.isinf([val for val in expected_moments.values()]).any()):
      raise exceptions.NumericalFittingError(
          routine='GEMA',
          condition=f'NaN or inf found on iteration {iteration}.',
      )

    # Step #2b: Weight with regional skew.
    # TODO(gsnearing) Implement regional skew.

    # Step #2c: Check for convergence.
    converged = _check_convergence_norm1(
        current_moments=expected_moments,
        previous_moments=previous_moments,
        convergence_threshold=convergence_threshold,
    )
    previous_moments = expected_moments

    if iteration > _MAX_ITERATIONS:
      raise exceptions.NumericalFittingError(
          routine='GEMA',
          condition='max iterations reached'
      )

  return parameters


class GEMAFitter(base_fitter.BaseFitter):
  """Estimates return periods using the Generalized Expected Moments Algorithm.

  This is the baseline algorithm from Bulletin 17c.
  """

  def __init__(
      self,
      data: np.ndarray,
      kn_table: Optional[Mapping[int, float]] = None,
      convergence_threshold: Optional[float] = None,
      log_transform: bool = True
  ):
    """Constructor for a GEMA distribution fitter.

    Fits parameters of a log-Pearson-III distribution with the iterative EMA
    procedure, using the generalized versions of the distribution moments
    and interval moments.

    Args:
      data: Flow peaks to fit in physical units.
      kn_table: Custom test statistics table to override the Kn Table from
        Bulletin 17b in the Standard Grubbs Beck Test (GBT).
      convergence_threshold: Convergence threshold to be applied to the first
        norm of the moments of the EMA-estimated distribution. The default value
        is 1e-10.
      log_transform: Whether to transform the data before fitting a
        distribution.
    """
    super().__init__(
        data=data,
        log_transform=log_transform
    )

    # Find the PILF threshold.
    # TODO(gsnearing): Use Multiple Grubbs-Beck Test instead of Grubbs-Beck
    # Test.
    self._pilf_tester = gbt.GrubbsBeckTester(
        data=self.transformed_sample,
        kn_table=kn_table,
    )

    # Run the EMA algorithm.
    self._distribution_parameters = _ema(
        systematic_record=self.transformed_sample,
        pilf_threshold=self._pilf_tester.pilf_threshold,
        convergence_threshold=convergence_threshold,
    )

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
    return 1 - tdu.pearson3_cdf(
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
    transformed_flow_values = tdu.pearson3_invcdf(
        alpha=self._distribution_parameters['alpha'],
        beta=self._distribution_parameters['beta'],
        tau=self._distribution_parameters['tau'],
        quantiles=(1 - exceedance_probabilities),
    )
    return self._untransform_data(data=transformed_flow_values)
