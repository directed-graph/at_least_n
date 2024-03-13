import at_least_n

import dataclasses
import itertools
import unittest

from typing import Iterable


@dataclasses.dataclass
class Parameters:
  probabilities: Iterable[float]
  n: int
  expected_probability: float


class TestAtLeastN(unittest.TestCase):
  """Unit tests for the at_least_n module.

  Run with: `python -m unittest --verbose`.
  """
  parameters = [
      Parameters([0], 2, 0.0),
      Parameters([1], 2, 0.0),
      Parameters([1], 1, 1.0),
      Parameters([0.5], 1, 0.5),
      Parameters([1, 0], 1, 1.0),
      Parameters([1, 0], 2, 0.0),
      Parameters([1, 1], 1, 1.0),
      Parameters([0.5, 0.5, 0.5], 2, 0.5),
      Parameters([0.5, 0.5, 0.5], 3, 0.125),
      Parameters([1, 0, 0], 1, 1.0),
      Parameters([1, 0, 0], 2, 0.0),
      Parameters([1, 1, 0], 1, 1.0),
      Parameters([1, 1, 1], 1, 1.0),
      Parameters([0.5, 0.5, 0.5, 0.5], 1, 0.9375),
      Parameters([1, 1, 1, 0.5], 4, 0.5),
      Parameters([1, 1, 1, 0.5, 0.5], 5, 0.25),
      Parameters([1, 1, 1, 0.5, 0.5, 0], 5, 0.25),
      Parameters([1, 1, 1, 0.5, 0.5, 0.5], 5, 0.5),
      Parameters([1, 1, 1, 0.9, 0.2, 0], 5, 0.18),
      Parameters([1, 1, 1, 0.9, 0.2, 0.1], 5, 0.254),
  ]

  def _test_parameter(self, parameter: Parameters) -> None:
    """Tests at_least_n with the given parameter."""
    self.assertAlmostEqual(
        at_least_n.at_least_n(parameter.probabilities, parameter.n),
        parameter.expected_probability)

  def test_parameterized(self) -> None:
    """Verifies all items in `parameters` are correct."""
    for parameter in self.parameters:
      with self.subTest(parameter):
        self._test_parameter(parameter)

  def test_parameterized_permutated(self) -> None:
    """Verifies all permutations of items in `parameters` are correct."""
    for parameter in self.parameters:
      for probabilities in set(itertools.permutations(parameter.probabilities)):
        sub_parameter = Parameters(**dataclasses.asdict(parameter))
        sub_parameter.probabilities = probabilities
        with self.subTest(sub_parameter):
          self._test_parameter(sub_parameter)

  def test_parameterized_permutated_as_iter(self) -> None:
    """Verifies all permutations of iter items in `parameters` are correct."""
    for parameter in self.parameters:
      for probabilities in set(itertools.permutations(parameter.probabilities)):
        sub_parameter = Parameters(**dataclasses.asdict(parameter))
        sub_parameter.probabilities = iter(probabilities)
        with self.subTest(sub_parameter):
          self._test_parameter(sub_parameter)

  def test_n_invalid(self) -> None:
    """Verifies that invalid n is handled."""
    parameters = [0, -1, -10]
    for n in parameters:
      with self.subTest(n):
        with self.assertRaisesRegex(ValueError, 'n must be greater than 0'):
          at_least_n.at_least_n([], n)

  def test_empty_probabilities(self) -> None:
    """Verifies that empty probabilities are handled."""
    self.assertEqual(at_least_n.at_least_n([], 10), 0.0)


if __name__ == '__main__':
  unittest.main()
