import compute

import dataclasses
import enum
import math
import unittest

from typing import Any, Iterable, Iterator, Mapping


@dataclasses.dataclass
class SetPropertiesParameter:
  name: str
  input_properties: Mapping[str, Any]
  expected_properties: Mapping[str, Any]


@dataclasses.dataclass
class EvaluateParameter:
  name: str
  input_properties: Mapping[str, Any]
  expected_results: Iterator[compute.NameProbabilityPair]


class MyAttribute(enum.Enum):
  A_0 = enum.auto()
  A_1 = enum.auto()
  A_2 = enum.auto()


_MY_ATTRIBUTES_BY_NAME: compute.Dataset = {
    'name_0': {
        MyAttribute.A_0: 0.5,
        MyAttribute.A_1: 0.75,
        MyAttribute.A_1: 0.5,
    },
    'name_1': {
        MyAttribute.A_0: 0.99,
        MyAttribute.A_1: 0,
        MyAttribute.A_2: 0.5,
    },
}

_MY_THRESHOLD = 0.6

_MY_ROUND_TO = 5

_MY_N = 10

_MY_DEFAULT_ATTRIBUTES = {}

_MY_DEFAULT_PROBABILITY = 0.1


def _make_attributes(
    probabilities: Iterable[compute.Probability]) -> compute.Attributes:
  attribute_names = iter(MyAttribute)
  attributes: compute.Attributes = {}
  for probability in probabilities:
    attributes[next(attribute_names)] = probability
  return attributes


class TestCompute(unittest.TestCase):
  """Unit tests for the compute module.

  Run with: `python -m unittest --verbose`.
  """

  _SET_PROPERTIES_TESTS: Iterable[SetPropertiesParameter] = [
      SetPropertiesParameter(
          'set_all', {
              'attributes_by_name': _MY_ATTRIBUTES_BY_NAME,
              'AttributeType': MyAttribute,
              'threshold': _MY_THRESHOLD,
              'n': _MY_N,
              'round_to': _MY_ROUND_TO,
              'default_attributes': _MY_DEFAULT_ATTRIBUTES,
              'default_probability': _MY_DEFAULT_PROBABILITY,
          }, {
              'attributes_by_name': _MY_ATTRIBUTES_BY_NAME,
              'AttributeType': MyAttribute,
              'threshold': _MY_THRESHOLD,
              'n': _MY_N,
              'round_to': _MY_ROUND_TO,
              'default_attributes': _MY_DEFAULT_ATTRIBUTES,
              'default_probability': _MY_DEFAULT_PROBABILITY,
          }),
      SetPropertiesParameter(
          'set_threshold', {
              'attributes_by_name': _MY_ATTRIBUTES_BY_NAME,
              'AttributeType': MyAttribute,
              'threshold': _MY_THRESHOLD,
          }, {
              'attributes_by_name': _MY_ATTRIBUTES_BY_NAME,
              'AttributeType': MyAttribute,
              'threshold': _MY_THRESHOLD,
              'n': math.ceil(len(MyAttribute) * _MY_THRESHOLD),
          }),
      SetPropertiesParameter(
          'no_set_default_attributes', {
              'attributes_by_name': _MY_ATTRIBUTES_BY_NAME,
              'AttributeType': MyAttribute,
          }, {
              'attributes_by_name': _MY_ATTRIBUTES_BY_NAME,
              'AttributeType': MyAttribute,
              'default_attributes': {},
          }),
  ]

  def test_set_properties(self) -> None:
    """Verifies properties are properly set in initializer."""
    for parameter in self._SET_PROPERTIES_TESTS:
      evaluator = compute.Evaluator(**parameter.input_properties)
      with self.subTest(parameter.name):
        self.assertDictEqual(
            {
                key: getattr(evaluator, key)
                for key in parameter.expected_properties.keys()
            }, parameter.expected_properties)

  _EVALUATE_TESTS: Iterable[EvaluateParameter] = [
      EvaluateParameter(
          'no_missing', {
              'attributes_by_name': {
                  'name_0': _make_attributes([0.5, 0.5, 0.5]),
                  'name_1': _make_attributes([0.2, 0.7, 0.9]),
                  'name_2': _make_attributes([0.5, 0.1, 0.5]),
              },
              'AttributeType': MyAttribute,
              'n': 3,
          }, [
              ('name_1', 0.126),
              ('name_0', 0.125),
              ('name_2', 0.025),
          ]),
      EvaluateParameter(
          'default_probability', {
              'attributes_by_name': {
                  'name_0': _make_attributes([0.5, 0.5, 0.5]),
                  'name_1': _make_attributes([0.2]),
                  'name_2': _make_attributes([]),
              },
              'AttributeType': MyAttribute,
              'n': 3,
              'default_probability': 1.0,
          }, [
              ('name_2', 1.0),
              ('name_1', 0.2),
              ('name_0', 0.125),
          ]),
      EvaluateParameter(
          'default_attributes', {
              'attributes_by_name': {
                  'name_0': _make_attributes([0.5, 0.5, 0.5]),
                  'name_1': _make_attributes([0.25]),
                  'name_2': _make_attributes([]),
              },
              'AttributeType': MyAttribute,
              'n': 1,
              'default_probability': 1.0,
              'default_attributes': _make_attributes([0.0, 0.0, 0.0]),
          }, [
              ('name_0', 0.875),
              ('name_1', 0.25),
              ('name_2', 0.0),
          ]),
  ]

  def test_evaluate(self) -> None:
    """Verifies evaluate works for different configurations."""
    for parameter in self._EVALUATE_TESTS:
      with self.subTest(parameter.name):
        self.assertEqual(
            list(compute.Evaluator(**parameter.input_properties).evaluate()),
            parameter.expected_results)

  def test_print(self) -> None:
    """Verifies the __str__ printing works as expected."""
    properties = {
        'attributes_by_name': {
            'n_0': _make_attributes([0.5, 0.5, 0.5]),
            'name_1': _make_attributes([0.2, 0.7, 0.9]),
            'name2': _make_attributes([0.5, 0.1, 0.5]),
        },
        'AttributeType': MyAttribute,
        'n': 1,
        'round_to': 5,
    }
    expected_output = '\n'.join([
        'name_1 0.97600',
        '   n_0 0.87500',
        ' name2 0.77500',
    ])
    self.assertEqual(str(compute.Evaluator(**properties)), expected_output)


if __name__ == '__main__':
  unittest.main()
