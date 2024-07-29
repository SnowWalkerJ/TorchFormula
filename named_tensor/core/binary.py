from typing import Tuple
import torch as th
from .tensor import NamedTensor


def _align(item1: NamedTensor, item2: NamedTensor) -> Tuple[NamedTensor, NamedTensor]:
  dims = item1.dims.copy()
  dims_set = set(dims)
  dims.extend([dim for dim in item2.dims if dim not in dims_set])
  dims1 = set(item1.dims)
  dims2 = set(item2.dims)
  strides1 = []
  strides2 = []
  coords = {}
  shape = []
  for dim in dims:
    if dim not in dims1:
      strides1.append(0)
      strides2.append(item2.stride(dim))
      coords[dim] = item2.coords[dim]
      shape.append(len(coords[dim]))
    elif dim not in dims2:
      strides1.append(item1.stride(dim))
      strides2.append(0)
      coords[dim] = item1.coords[dim]
      shape.append(len(coords[dim]))
    else:
      if item1.coords[dim] != item2.coords[dim]:
        raise ValueError(f"coords not aligned: `{dim}`")
      strides1.append(item1.stride(dim))
      strides2.append(item2.stride(dim))
      coords[dim] = item1.coords[dim]
      shape.append(len(coords[dim]))

  item1 = NamedTensor(th.as_strided(item1.data, shape, strides1), dims, coords)
  item2 = NamedTensor(th.as_strided(item2.data, shape, strides2), dims, coords)
  return item1, item2


def _binary(op, item1: NamedTensor, item2: NamedTensor, *args, **kwargs) -> NamedTensor:
  if isinstance(item1, NamedTensor) and isinstance(item2, NamedTensor):
    item1, item2 = _align(item1, item2)
  return NamedTensor(op(item1.data, item2.data, *args, **kwargs), item1.dims, item1.coords)


def add(item1: NamedTensor, item2: NamedTensor, *, alpha=1):
  return _binary(th.add, item1, item2, alpha=alpha)


def sub(item1: NamedTensor, item2: NamedTensor, *, alpha=1):
  return _binary(th.sub, item1, item2, alpha=alpha)


def mul(item1: NamedTensor, item2: NamedTensor):
  return _binary(th.mul, item1, item2)


def div(item1: NamedTensor, item2: NamedTensor):
  return _binary(th.div, item1, item2)


def pow(item1: NamedTensor, item2: NamedTensor):
  return _binary(th.pow, item1, item2)

