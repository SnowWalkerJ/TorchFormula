from typing import ClassVar, Callable
import torch as th
from named_tensor import NamedTensor
from ..core.formula import TorchFormula
from ..core.module import TFModule


class ReduceFormula(TorchFormula):
  def __init__(self, base, operation, dim):
    self.base = base
    self.operation = operation
    self.dim = dim
    super().__init__()

  def get_dependency(self):
    return {"base": self.base}

  def compile(self, deps):
    base: NamedTensor = deps["base"]
    if self.dim is None:
      self.variable['axis'] = None
      dims = []
      coords = {}
    else:
      self.variable['axis'] = base.dims.index(self.dim)
      dims = [_dim for _dim in base.dims if _dim != self.dim]
      coords = {_dim: coord for _dim, coord in base.coords.items() if _dim != self.dim}
    return None, NamedTensor(self.compute({"base": base.data}, None), dims, coords)

  def compute(self, deps, module):
    return self.operation(deps['base'], self.variable['axis'])


class ReduceModule(TFModule):
  operation: ClassVar[Callable[[th.Tensor], th.Tensor]]

  def __init__(self, dim):
    self.dim = dim

  def __call__(self, tensor):
    return ReduceFormula(tensor, self.operation, self.dim)


class Mean(ReduceModule):
  operation = th.mean


class Sum(ReduceModule):
  operation = th.sum


def mean(tensor, dim=None):
  return Mean(dim)(tensor)


def sum(tensor, dim=None):
  return Sum(dim)(tensor)
