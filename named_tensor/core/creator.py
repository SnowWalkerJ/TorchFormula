import xarray as xr
import torch as th
from .tensor import NamedTensor


__all__ = [
  "full", "zeros", "ones", "zeros_like", "ones_like", "arange",
]


def full(value, dtype, dims, coords, attrs=None, real=False, device="cpu") -> NamedTensor:
  shape = [len(coords[dim]) for dim in dims]
  if real:
    data = th.full(shape, value, dtype=dtype, device=device)
  else:
    data = th.as_strided(th.tensor([value], dtype=dtype, device=device), shape, [0] * len(dims))
  return NamedTensor(data, dims, coords, attrs=attrs)


def zeros(dtype, dims, coords, attrs=None, real=False, device="cpu") -> NamedTensor:
  return full(0, dtype, dims, coords, attrs, real, device=device)


def ones(dtype, dims, coords, attrs=None, real=False, device="cpu") -> NamedTensor:
  return full(1, dtype, dims, coords, attrs, real, device=device)


def zeros_like(data: NamedTensor, real=False, device="cpu") -> NamedTensor:
  return zeros(data.dtype, data.dims, data.coords, data.attrs, real, device=device)


def ones_like(data: NamedTensor, real=False, device="cpu") -> NamedTensor:
  return ones(data.dtype, data.dims, data.coords, data.attrs, real, device=device)


def from_xarray(data: xr.DataArray, device="cpu") -> NamedTensor:
  return NamedTensor(th.from_numpy(data.values).to(device, copy=False), data.dims, data.coords, data.attrs)


def arange(dim: str, *args, dtype=None, device=None, attrs=None) -> NamedTensor:
  data = th.arange(*args, dtype=dtype, device=device)
  coords = list(range(len(data)))
  return NamedTensor(data, [dim], {dim: coords}, attrs=attrs)
