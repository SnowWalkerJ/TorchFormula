from typing import List, Optional, Tuple, Union
import torch as th
import xarray as xr


class NamedTensor:
  data: th.Tensor
  dims: List[str]
  coords: dict
  attrs: Optional[str]

  def __init__(self, data: th.Tensor, dims: List[str], coords: dict, attrs=None):
    self.data = data
    if len(dims) != len(data.shape):
      raise ValueError("len(dims) != len(data.shape)")
    for dim, size in zip(dims, data.shape):
      if len(coords[dim]) != size:
        raise ValueError(f"coords of {dim} not aligned with data size: {len(coords[dim])} != {size}")
    self.dims = dims
    self.coords = coords
    self.attrs = attrs

  @property
  def dtype(self) -> th.dtype:
      return self.data.dtype

  @property
  def shape(self) -> th.Size:
    return self.data.shape

  @property
  def strides(self) -> Tuple[int, ...]:
    return self.stride()

  @property
  def device(self) -> th.device:
      return self.data.device

  def stride(self, dim=None) -> Union[int, Tuple[int, ...]]:
    return self.data.stride(self.dims.index(dim))

  def sum(self, dim=None):
    from .reduction import sum
    return sum(self, dim)

  def mean(self, dim=None):
    from .reduction import mean
    return mean(self, dim)

  def var(self, dim=None):
    from .reduction import var
    return var(self, dim)

  def std(self, dim=None):
    from .reduction import std
    return std(self, dim)

  def max(self, dim=None):
    from .reduction import max
    return max(self, dim)

  def min(self, dim=None):
    from .reduction import min
    return min(self, dim)

  def squeeze(self, dim):
    from .reduction import _reduce
    axis = self.dims.index(dim)
    if self.shape[axis] == 1:
      return _reduce(self, "squeeze", dim)
    else:
      return self

  def transpose(self, *dims):
    un_assigned_dims = set(self.dims) - set(dims)
    if non_exist_dims := set(dims) - set(self.dims):
      raise ValueError(f"transposing non-exist dims: {non_exist_dims}")
    out_dims = []
    for dim in dims:
      if dim is Ellipsis:
        out_dims.extend([dim for dim in self.dims if dim in un_assigned_dims])
      else:
        out_dims.append(dim)
    if set(out_dims) != set(self.dims):
      raise ValueError
    new_axis = [self.dims.index(dim) for dim in out_dims]
    return NamedTensor(self.data.permute(new_axis), out_dims, self.coords)

  def detach(self):
    return NamedTensor(self.data.detach(), self.dims, self.coords)

  def to(self, device):
    return NamedTensor(self.data.to(device), self.dims, self.coords)

  def set_index(self, **indices):
    new_coords = self.coords.copy()
    for dim, coord in indices.items():
      if len(coord) != len(self.coords[dim]):
        raise ValueError(f"new coord length not aligned with data ({dim}): {len(coord)} != {len(self.coords[dim])}")
      new_coords[dim] = coord
    return NamedTensor(self.data, self.dims, new_coords, attrs=self.attrs)

  def astype(self, dtype):
    from .unary import astype
    return astype(self, dtype)

  def to_tensor(self):
    return self.data

  def to_numpy(self):
    return self.data.numpy()

  def to_xarray(self):
    return xr.DataArray(self.data.cpu().numpy(), dims=self.dims, coords=self.coords)

  def item(self):
    return self.data.item()

  def __abs__(self):
    from .unary import abs
    return abs(self)

  def __neg__(self):
    from .unary import neg
    return neg(self)

  def __pos__(self):
    return self

  def __add__(self, other):
    from .binary import add
    return add(self, other)

  def __radd__(self, other):
    from .binary import add
    return add(other, self)

  def __sub__(self, other):
    from .binary import sub
    return sub(self, other)

  def __rsub__(self, other):
    from .binary import sub
    return sub(other, self)

  def __mul__(self, other):
    from .binary import mul
    return mul(self, other)

  def __rmul__(self, other):
    from .binary import mul
    return mul(other, self)

  def __truediv__(self, other):
    from .binary import div
    return div(self, other)

  def __rtruediv__(self, other):
    from .binary import div
    return div(other, self)
