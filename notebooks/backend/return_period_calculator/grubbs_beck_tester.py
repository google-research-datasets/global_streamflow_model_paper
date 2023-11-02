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

"""Implements utilities for handling Potentially Impactful Low Floods (PILFs).

This procedure for handling PILFs is described in Appendix 4 of Bulletin 17b.
It was supersceded by the Multiple Grubbs-Beck Test in Appendix 6 of Bulletin
17c.

This code was developed as part of an open source Python package for
calcultaing streamflow return periods according to guidelines in the USGS
Bulletin 17c (2019):

https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""
import abc
import logging
from typing import Optional, Mapping

import numpy as np
import pandas as pd

from backend.return_period_calculator import exceptions

_KN_TABLE_FILENAME = './backend/return_period_calculator/bulletin17b_kn_table.csv'  # pylint: disable=line-too-long


def _load_kn_table(
    file: str = _KN_TABLE_FILENAME
) -> Mapping[int, float]:
  kn_table_series = pd.read_csv(file, index_col='Sample Size')
  kn_table_series.index = kn_table_series.index.astype(int)
  return kn_table_series.to_dict()['KN Value']


def _grubbs_beck_test(
    sorted_data: np.ndarray,
    kn_table: Mapping[int, float],
) -> int:
  """Performs one sweep of a GBT.

  This routine implements Equation 8a in the USGS Bulletin 17b (not 17c):
  https://water.usgs.gov/osw/bulletin17b/dl_flow.pdf

  Cohn et al (2013) argue that a multiple Grubbs Beck Test should be used
  instead.

  Cohn, T. A., et al. "A generalized Grubbs-Beck test statistic for detecting
  multiple potentially influential low outliers in flood series."
  Water Resources Research 49.8 (2013): 5047-5058.

  Args:
    sorted_data: Sample to test.
    kn_table: Mapping of the pre-calculated test statistic table from Appendix
      4. Keys in this mapping are sample size and values are the test statistic
      at that sample size.

  Returns:
    Index of the first discarded in the sorted array.

  Raises:
    NotEnoughDataError if the KN table does not support the number of samples.
  """
  # Under normal conditions, we will check up to half of the data as possible
  # low outliers. For short data records, this is not possible (the minimum
  # sample size for GBT is usually 10, depending on the KN table). For sample
  # sizes between this miniumm (e.g., 10) and double the minimum (e.g., 20), we
  # will check as many as we can.
  min_sample_size = min(kn_table.keys())
  num_samples = len(sorted_data)
  if num_samples <= min_sample_size or num_samples > max(kn_table.keys()):
    raise exceptions.NotEnoughDataError(
        routine='Grubbs-Beck test KN table',
        num_data_points=num_samples,
        data_requirement=min(kn_table.keys())+1,
    )
  max_sample_position_to_test = min(
      [int(num_samples / 2), num_samples - min_sample_size - 1])

  # Do not remove more than half of the data.
  for k in range(max_sample_position_to_test, 0, -1):

    # Calculate lower threshold. This uses a statistics table that was
    # calculated for a 10% confidence threshold (pg 4-1 in Bulletin 17b).
    mu_remove_k = sorted_data[k+1:].mean()
    sigma_remove_k = sorted_data[k+1:].std()
    num_samples = len(sorted_data[k+1:])
    lower_threshold = mu_remove_k - kn_table[num_samples] * sigma_remove_k

    # Return index of the largest rejected sample.
    if sorted_data[k] < lower_threshold:
      return k

  # If no outliers are detected.
  return -1


class GrubbsBeckTester(abc.ABC):
  """Grubbs-Beck Test object.

  All this object does is store the values that we might need from n GBT.

  Attributes:
    pilf_threshold: (log-transformed) flow value such that anything below this
      value is considered a potentially impactful low flood.
    in_population_sample: Portion of the original sample that are not PILFs.
    pilf_sample: Portion of the original sample that were discarded as PILFs.
  """

  def __init__(
      self,
      data: np.ndarray,
      kn_table: Optional[Mapping[int, float]] = None,
  ):
    """Constructor for a GBT object.

    Args:
      data: Sample to test.
      kn_table: Option to load in a pre-defined table of Kn test statistics
        instead of reading the default table, which calculates everything at a
        10% confidence interval. Keyed by sample size (integers).
    """
    # Load a test statistics table if one is not provided.
    if kn_table is None:
      kn_table = _load_kn_table()

    # Run the test on log-transformed data. If the test fails in any way,
    # we resort to fitting the theoretical distribution with all data, which
    # means that EMA is effectively identical to simple distribution fitting.
    sorted_data = np.sort(data)
    try:
      pilf_index = _grubbs_beck_test(
          sorted_data=sorted_data,
          kn_table=kn_table,
      )
    except exceptions.NotEnoughDataError:
      logging.exception('Not enough data for Grubbs-Beck test, resorting to '
                        'assuming no outliers.')
      pilf_index = -1

    # Separate the sample.
    if pilf_index < 0:
      self._in_pop_sample = sorted_data
      self._out_of_pop_sample = np.array([])
      # This small threshold ensures that any zero flows are caught as PILFS.
      self._threshold = min(sorted_data) / 2
    else:
      self._in_pop_sample = sorted_data[pilf_index:]
      self._out_of_pop_sample = sorted_data[:pilf_index]
      # Threshold as the median between highest PILF and lowest non-PILF.
      # Return the PILF threshold in physical (not log-transformed) units.
      self._threshold = (max(self._out_of_pop_sample) +
                         min(self._in_pop_sample)) / 2

  @property
  def pilf_threshold(self) -> float:
    """Value of the PILF threshold as estimated by a GBT."""
    return self._threshold

  @property
  def in_population_sample(self) -> np.ndarray:
    """Portion of the data record that are not PILFs."""
    return self._in_pop_sample

  @property
  def pilf_sample(self) -> np.ndarray:
    """Portion of the data record that are PILFs."""
    return self._out_of_pop_sample



