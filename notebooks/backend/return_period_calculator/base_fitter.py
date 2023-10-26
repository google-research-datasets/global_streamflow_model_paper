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

"""Base class for different options for fitting return period distributions.

This code implements flood frequency interval calculations from guidelines
in the USGS Bulletin 17c (2019):

https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""
import abc
from typing import Optional

import numpy as np

from backend.return_period_calculator import exceptions

_MIN_DATA_POINTS = 2


def test_for_out_of_bounds_probabilities(
    probabilities: np.ndarray
):
  """Tests whether probabilities are in (0, 1) noninclusive."""
  mask = (probabilities <= 0) | (probabilities >= 1)
  if mask.any():
    bad_values = probabilities[mask]
    raise ValueError('Cumulative probabilitities must be in (0, 1), '
                     f'received the following bad values: {bad_values}.')


class BaseFitter(abc.ABC):
  """Base class for classes of fitters for return period distributions."""

  def __init__(
      self,
      data: np.ndarray,
      log_transform: bool = True
  ):
    """Common constructor for fitters.

    Args:
      data: Flow peaks to fit in physical units.
      log_transform: Whether to transform the data before fitting a
        distribution.

    Raises:
      NotEnoughDataError if the sample size is too small.
    """

    self._log_transform = log_transform
    self._transformed_data = self._transform_data(data)
    num_nonzeros = len(self._transformed_data)
    self._zeros_data = np.zeros(len(data) - num_nonzeros)
    if num_nonzeros < _MIN_DATA_POINTS:
      raise exceptions.NotEnoughDataError(
          num_data_points=num_nonzeros,
          data_requirement=_MIN_DATA_POINTS,
          routine=self.__class__.__name__,
      )
    self._pilf_threshold = min(self._transformed_data) / 2

  @property
  def type_name(self) -> str:
    return 'Base Fitter'

  def _transform_data(self, data: np.ndarray) -> np.ndarray:
    if self._log_transform:
      return np.log10(data[data > 0])
    else:
      return data[data > 0]

  def _untransform_data(self, data: np.ndarray) -> np.ndarray:
    if self._log_transform:
      return 10**data
    else:
      return data

  def _convert_exceedance_probabilities_to_return_periods(
      self,
      exceedance_probabilities: np.ndarray
  ) -> np.ndarray:
    if ((exceedance_probabilities <= 0).any() or
        (exceedance_probabilities >= 1).any()):
      raise ValueError('Probabilities are out of range.')
    return 1/exceedance_probabilities

  def _convert_return_periods_to_exceedance_probabilities(
      self,
      return_periods: np.ndarray
  ) -> np.ndarray:
    if (return_periods <= 0).any():
      raise ValueError('Return periods must be positive.')
    # Return periods might come in as integers, so need to check that they are
    # floats.
    return 1/return_periods.astype(float)

  def return_periods_from_flow_values(
      self,
      flows: np.ndarray,
  ) -> np.ndarray:
    """Predicts return periods from streamflow values.

    Args:
      flows: Streamflow values in physical units.

    Returns:
      Return periods predicted for each flow value.
    """
    exceedance_probabilities = self.exceedance_probabilities_from_flow_values(
        flows)
    return self._convert_exceedance_probabilities_to_return_periods(
        exceedance_probabilities)

  def flow_values_from_return_periods(
      self,
      return_periods: np.ndarray,
  ) -> np.ndarray:
    """Predicts streamflow values from return periods.

    Args:
      return_periods: Return periods from which to calculate flow values.

    Returns:
      Predicted flow values corresponding to requeseted return_periods.
    """
    exceedance_probabilities = self._convert_return_periods_to_exceedance_probabilities(
        return_periods)
    return self.flow_values_from_exceedance_probabilities(
        exceedance_probabilities)

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
    raise NotImplementedError()

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
    raise NotImplementedError()

  def probability_mass_function(
      self,
      edges: np.ndarray,
  ) -> np.ndarray:
    """Returns the (implied) probability mass function for a fitter."""
    cumulative_probabilities = 1 - self.exceedance_probabilities_from_flow_values(
        flows=edges)
    return np.diff(cumulative_probabilities)

  @property
  def pilf_threshold(self) -> Optional[float]:
    """Value of the PILF threshold as estimated by a GBT."""
    return self._untransform_data(self._pilf_threshold)[0]

  @property
  def in_population_sample(self) -> Optional[np.ndarray]:
    """Portion of the data record that are not PILFs."""
    if hasattr(self, '_pilf_tester'):
      return np.sort(self._untransform_data(
          self._pilf_tester.in_population_sample))
    else:
      return np.sort(self._untransform_data(self._transformed_data))

  @property
  def pilf_sample(self) -> Optional[np.ndarray]:
    """Portion of the data record that are PILFs."""
    if hasattr(self, '_pilf_tester'):
      return np.sort(self._untransform_data(self._pilf_tester.pilf_sample))
    else:
      return np.array([])

  @property
  def zeros_sample(self) -> np.ndarray:
    return self._zeros_data

  @property
  def transformed_sample(self) -> np.ndarray:
    return np.sort(self._transformed_data)

  @property
  def non_transformed_sample(self) -> np.ndarray:
    return self._untransform_data(self.transformed_sample)
