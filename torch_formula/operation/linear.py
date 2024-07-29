from typing import Optional
import torch as th
import torch.nn as nn
from named_tensor import NamedTensor
from ..core.formula import TorchFormula
from ..core.module import TFModule


class LinearFormula(TorchFormula):
  def __init__(self, formula, module):
    self.formula = formula
    self.module = module
    super().__init__()

  @property
  def dim(self):
    return self.module.dim

  @property
  def out_size(self):
    return self.module.out_size

  def get_dependency(self):
    return {"base": self.formula}

  def compile_offline(self, deps):
    base = deps["base"]
    self.variable['axis'] = base.dims.index(self.dim)
    in_size = base.shape[self.variable['axis']]
    module = self.module.get_module(in_size)
    out_dims = [_dim for _dim in base.dims if _dim != self.dim] + [self.dim]
    coords = base.coords.copy()
    coords[self.dim] = list(range(self.out_size))
    shape = [s for s, _dim in zip(base.shape, base.dims) if _dim != self.dim] + [self.out_size]
    out_data = th.ones(shape, dtype=base.data.dtype, device=base.data.device)
    return module, NamedTensor(out_data, out_dims, coords)

  def compute_offline(self, deps, module):
    x = deps["base"]
    x = x.moveaxis(self.variable['axis'], -1)
    x = module(x)
    return x

  def initialize(self, deps, module):
    base = deps["base"]
    self.variable['axis'] = base.dims.index(self.dim)
    out_dims = [_dim for _dim in base.dims if _dim != self.dim] + [self.dim]
    coords = base.coords.copy()
    coords[self.dim] = list(range(self.out_size))
    shape = [s for s, _dim in zip(base.shape, base.dims) if _dim != self.dim] + [self.out_size]
    out_data = th.ones(shape, dtype=base.data.dtype, device=base.data.device)
    return NamedTensor(out_data, out_dims, coords)

  def compute_online(self, deps, module):
    raise NotImplementedError


class Linear(TFModule):
  def __init__(self, dim, out_size, in_size=None):
    self.dim = dim
    self.out_size = out_size
    self.in_size = in_size
    self.module = None

  def __call__(self, x):
    return LinearFormula(x, self)

  def get_module(self, in_size: int) -> nn.Module:
    if self.in_size is not None and in_size != self.in_size:
      raise ValueError
    self.in_size = in_size
    if self.module is None:
      self.module = nn.Linear(self.in_size, self.out_size)
    return self.module


def linear(formula: TorchFormula, dim: Optional[str], out_size: int) -> TorchFormula:
  module = Linear(dim, out_size)
  return LinearFormula(formula, module)
