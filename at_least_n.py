import functools
import logging

from typing import Callable, Iterable


class AtLeastN:
  """Computes the probability of at least n resolves to true.

  Given a list of N probabilities, computes the probability that at least n < N
  of them resolves to true.

  This is implemented as a class so we can have a "global" probabilities list.
  """

  def __init__(self, probabilities: Iterable[float]):
    self.probabilities = list(probabilities)

  def compute(self, n: int) -> float:
    """Computes the probability that at least n > 0 items resolve to true."""
    if n < 1:
      raise ValueError('n must be greater than 0')

    return self._n_or_more(len(self.probabilities) - 1, n)

  def _zero_condition(self, m: int, n: int) -> bool:
    """Returns true if these parameters should return probability 0.0."""
    # m + 1 means number of items in the list; if that is less than n, then
    # there are not enough elements to resolve at least n items.
    not_enough_elements = m + 1 < n

    # If there are no elements in the list, then we cannot resolve any.
    no_elements_left = m < 0

    return not_enough_elements or no_elements_left

  @functools.lru_cache
  def _all_zero(self, m: int) -> float:
    """Returns the probability that all items 0 to m all resolve to false."""
    if m == 0:
      return 1.0 - self.probabilities[m]

    return functools.reduce(lambda p, q: p * (1.0 - q),
                            self.probabilities[:m + 1], 1.0)

  @functools.lru_cache
  def _at_least_one(self, m: int) -> float:
    """Returns the probability that at least one from 0 to m resolves true."""
    return 1.0 - self._all_zero(m)

  @functools.lru_cache
  def _exactly_one(self, m: int) -> float:
    """Returns the probability that exactly one from 0 to m resolves true."""
    if m == 0:
      return self.probabilities[m]

    return self.probabilities[m] * self._all_zero(m - 1) + (
        1.0 - self.probabilities[m]) * self._exactly_one(m - 1)

  @functools.lru_cache
  def _exactly_n(self, m: int, n: int) -> float:
    """Returns the probability of exactly n resolving true.

    Args:
      m: The index to start looking from (backwards).
      n: The number of items to resolve to true.
    """
    if self._zero_condition(m, n):
      return 0.0

    if n == 1:
      return self._exactly_one(m)

    return self.probabilities[m] * self._exactly_n(m - 1, n - 1) + (
        1.0 - self.probabilities[m]) * self._exactly_n(m - 1, n)

  @functools.lru_cache
  def _n_or_more(self, m: int, n: int) -> float:
    """Returns the probability of at least n > 0 resolving to true.

    Args:
      m: The index to start looking from (backwards).
      n: The number of items to resolve to true.
    """
    if self._zero_condition(m, n):
      return 0.0

    if n == 1:
      return self._at_least_one(m)

    return self.probabilities[m] * self._exactly_n(
        m - 1, n - 1) + self._n_or_more(m - 1, n)


def at_least_n(probabilities: Iterable[float], n: int) -> float:
  return AtLeastN(probabilities).compute(n)
