"""Custom exceptions for return period calculator.

This code implements flood frequency interval calculations from guidelines
in the USGS Bulletin 17c (2019):

https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""
from typing import Optional


class NotEnoughDataError(Exception):
  """Raised when there is not enough data for a statistical method."""

  def __init__(
      self,
      num_data_points: int,
      data_requirement: int,
      routine: str = '',
      prefix: Optional[str] = None
  ):
    self.message = f'Not enough data -- {routine} requires {data_requirement} data points, but {num_data_points} were supplied.'
    if prefix is not None:
      self.message = f'{prefix} {self.message}'
    super().__init__(self.message)


class NumericalFittingError(Exception):
  """Raised when there is an error in numerical fitting."""

  def __init__(
      self,
      routine: str,
      condition: str,
      prefix: Optional[str] = None
  ):
    self.message = f'{routine} failed with condition: {condition}.'
    if prefix is not None:
      self.message = f'{prefix} {self.message}'
    super().__init__(self.message)


class AskedForBackupError(Exception):
  """Raised to block primary methods when user asked for a backup method."""

  def __init__(
      self,
      method: str,
      prefix: Optional[str] = None
  ):
    self.message = f'User requested backup method {method}.'
    if prefix is not None:
      self.message = f'{prefix} {self.message}'
    super().__init__(self.message)
