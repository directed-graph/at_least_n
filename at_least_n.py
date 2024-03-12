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
    return self._n_or_more(len(self.probabilities) - 1, n)

  def _zero_condition(self, m: int, n: int) -> bool:
    """Returns true if these parameters should return probability 0.0."""
    # m + 1 means number of items in the list; if that is less than n, then
    # there are not enough elementsto resolve at least n items.
    not_enough_elements = m + 1 < n

    # If there are no elements in the list, then we cannot resolve any.
    no_elements_left = m == 0

    # If n is negative, then we don't want to consider this item. This exists
    # to calculate the case when n is 0.
    ignored = n < 0

    return not_enough_elements or no_elements_left or ignored

  @functools.lru_cache
  def _all_zero(self, m: int) -> float:
    """Returns the probability of the first m items resolving false."""
    return functools.reduce(lambda p, q: p * (1.0 - q), self.probabilities[:m],
                            1.0)

  @functools.lru_cache
  def _exactly_n(self, m: int, n: int) -> float:
    """Returns the probability of exactly n resolving true.

    Args:
      m: The index to start looking from (backwards).
      n: The number of items to resolve to true.
    """
    if m == 0 and n == 1:
      return self.probabilities[m]

    if self._zero_condition(m, n):
      return 0.0

    if n == 1:
      return self.probabilities[m] * self._all_zero(m) + (
          1 - self.probabilities[m]) * self._exactly_n(m - 1, n)

    return self.probabilities[m] * self._exactly_n(
        m - 1, n - 1) + (1 - self.probabilities[m]) * self._exactly_n(m - 1, n)

  @functools.lru_cache
  def _n_or_more(self, m: int, n: int) -> float:
    """Returns the probability of n or more resolving to true.

    Args:
      m: The index to start looking from (backwards).
      n: The number of items to resolve to true.
    """
    if self._zero_condition(m, n):
      return 0.0

    if n < 1:
      return max(self.probabilities[:m])

    return self.probabilities[m] * self._exactly_n(
        m - 1, n - 1) + self._n_or_more(m - 1, n)


def at_least_n(probabilities: Iterable[float], n: int) -> float:
  return AtLeastN(probabilities).compute(n)


if __name__ == '__main__':
  print(at_least_n([1, 1], 1)) # currently a bug
