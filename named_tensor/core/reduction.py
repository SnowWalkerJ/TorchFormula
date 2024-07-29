from typing import Optional
from .tensor import NamedTensor


__all__ = ['sum', 'mean', 'var', 'std', 'max', 'min']


def _reduce(tensor: NamedTensor, op: str, dim: Optional[str] = None) -> NamedTensor:
  if dim is None:
    data = getattr(tensor.data, op)()
    dims = []
    coords = {}
  else:
    axis = tensor.dims.index(dim)
    data = getattr(tensor.data, op)(axis)
    dims = [_dim for _dim in tensor.dims if _dim != dim]
    coords = {key: value for key, value in tensor.coords.items() if key != dim}
  return NamedTensor(data, dims, coords)


def sum(tensor, dim):
  return _reduce(tensor, "sum", dim)


def mean(tensor, dim):
  return _reduce(tensor, "mean", dim)


def var(tensor, dim):
  return _reduce(tensor, "var", dim)


def std(tensor, dim):
  return _reduce(tensor, "std", dim)


def max(tensor, dim):
  return _reduce(tensor, "max", dim)


def min(tensor, dim):
  return _reduce(tensor, "min", dim)
