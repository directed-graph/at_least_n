import at_least_n

import dataclasses
import enum
import functools
import math
import operator

from typing import Iterable, Iterator, Mapping, Optional, Tuple, Type

Name = str
Probability = float
Attributes = Mapping[enum.Enum, Probability]

Dataset = Mapping[Name, Attributes]
NameProbabilityPair = Tuple[Name, Probability]

_DEFAULT_THRESHOLD = 0.8
_DEFAULT_ROUND_TO = 3
_DEFAULT_PROBABILITY = 0.5


class Evaluator:
  """Class to evaluate the probability of each item in the Dataset.

  To use, first define your own enum.Enum as AttributeType. Then, create a
  Dataset. We use our own Dataset type here to make this program completely
  standalone (i.e. no pandas/numpy dependency).

  Example:

      import compute
      import enum

      class MyAttribute(enum.Enum):
        A_0 = enum.auto()
        A_1 = enum.auto()

      attributes_by_name: compute.Dataset = {
          'name_0': {
              MyAttribute.A_0: 0.5,
              MyAttribute.A_1: 0.75,
          },
          'name_1': {
              MyAttribute.A_0: 0.99,
              MyAttribute.A_1: 0,
          },
      }

      print(compute.Evaluator(attributes_by_name, MyAttribute))
  """

  def __init__(self,
               attributes_by_name: Dataset,
               AttributeType: Type[enum.Enum],
               threshold: float = _DEFAULT_THRESHOLD,
               n: Optional[int] = None,
               round_to: int = _DEFAULT_ROUND_TO,
               default_attributes: Optional[Attributes] = None,
               default_probability: Probability = _DEFAULT_PROBABILITY):
    """Initializes the computation.

    Args:
      attributes_by_name: The Dataset to compute probabilities on.
      AttributeType: The Attributes being used in the Dataset. This tells us
                     how to set up the "feature vector" of each item in the
                     Dataset.
      threshold: Determines the n as the percentage of length of AttributeType.
      n: The n value; set based on threshold if not provided.
      round_to: The number of decimal places to round to for the output.
      default_attributes: The default probability to use if not set in the
                          item.
      default_probability: The default probability to use if not set in the
                           default_attributes.
    """
    self.attributes_by_name = attributes_by_name
    self.threshold = threshold
    self.round_to = round_to
    self.default_probability = default_probability
    self.default_attributes = default_attributes
    if self.default_attributes is None:
      self.default_attributes = {}

    self.AttributeType = AttributeType
    self.n = n or math.ceil(self.threshold * len(self.AttributeType))

  def _encode(self, attributes: Attributes) -> Iterable[Probability]:
    """Encodes attributes as probability list.

    If the given attributes don't have the attribute name in AttributeType,
    then default_attributes will be used. If the attribute name is not in
    default_attributes either, then default_probability will be used.

    Args:
      attributes: The attributes to encode.
    """
    probabilities = []

    for attribute_name in self.AttributeType:
      probabilities.append(
          attributes.get(
              attribute_name,
              self.default_attributes.get(attribute_name,
                                          self.default_probability)))

    return probabilities

  def _compute_each(self) -> Iterator[NameProbabilityPair]:
    """Computes probability of at least n for each item."""
    for name, attributes in self.attributes_by_name.items():
      yield name, at_least_n.at_least_n(self._encode(attributes), self.n)

  @functools.lru_cache
  def evaluate(self) -> Iterator[NameProbabilityPair]:
    """Computes the reverse sorted of at least n for each item."""
    yield from reversed(
        sorted(self._compute_each(), key=operator.itemgetter(1)))

  def _compute_prints(self) -> Iterator[str]:
    """Computes each line to print for at least n for each item."""
    max_name = max(map(len, self.attributes_by_name.keys()))
    for name, probability in self.evaluate():
      yield '{:>{max_name}} {:.{round_to}f}'.format(
          name,
          round(probability, self.round_to),
          max_name=max_name,
          round_to=self.round_to)

  @functools.lru_cache
  def __str__(self) -> str:
    return '\n'.join(self._compute_prints())
