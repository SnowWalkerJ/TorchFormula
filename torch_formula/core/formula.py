import abc
import torch.nn as nn


class TorchFormula(abc.ABC):
  def __init__(self):
    self.variable = {}

  @abc.abstractmethod
  def get_dependency(self):
    raise NotImplementedError

  @abc.abstractmethod
  def compile_offline(self, deps):
    raise NotImplementedError

  @abc.abstractmethod
  def compute_offline(self, deps, module):
    raise NotImplementedError

  @abc.abstractmethod
  def initialize(self, deps, module):
    raise NotImplementedError

  @abc.abstractmethod
  def compute_online(self, deps, module):
    raise NotImplementedError
