import torch as th
from .tensor import NamedTensor


__all__ = ["sin", "cos", "tan", "tanh", "sinh", "cosh", "asin", "acos", "atan", "exp", "log", "sqrt",
           "acosh", "asinh", "astype", "clip", "clamp", "softmax", "abs", "neg",
           ]


def _unary(tensor: NamedTensor, func, *args, **kwargs) -> NamedTensor:
    return NamedTensor(func(tensor.data, *args, **kwargs), tensor.dims, tensor.coords)


def abs(tensor) -> NamedTensor:
  return _unary(tensor, th.abs)


def neg(tensor) -> NamedTensor:
  return _unary(tensor, th.neg)


def sin(tensor) -> NamedTensor:
  return _unary(tensor, th.sin)


def cos(tensor) -> NamedTensor:
  return _unary(tensor, th.cos)


def tan(tensor) -> NamedTensor:
  return _unary(tensor, th.tan)


def tanh(tensor) -> NamedTensor:
  return _unary(tensor, th.tanh)


def sinh(tensor) -> NamedTensor:
  return _unary(tensor, th.sinh)


def cosh(tensor) -> NamedTensor:
  return _unary(tensor, th.cosh)


def acosh(tensor) -> NamedTensor:
  return _unary(tensor, th.acosh)


def atan(tensor) -> NamedTensor:
  return _unary(tensor, th.atan)


def asin(tensor) -> NamedTensor:
  return _unary(tensor, th.asin)


def acos(tensor) -> NamedTensor:
  return _unary(tensor, th.acos)


def asinh(tensor) -> NamedTensor:
  return _unary(tensor, th.asinh)


def exp(tensor) -> NamedTensor:
  return _unary(tensor, th.exp)


def log(tensor) -> NamedTensor:
  return _unary(tensor, th.log)


def sqrt(tensor) -> NamedTensor:
  return _unary(tensor, th.sqrt)


def clamp(tensor, min, max) -> NamedTensor:
  return _unary(tensor, th.clmp, min, max)


def clip(tensor, min, max) -> NamedTensor:
  return clamp(tensor, min, max)


def astype(tensor, dtype) -> NamedTensor:
  return _unary(tensor, th.astype, dtype)


def softmax(tensor, dim, dtype=None) -> NamedTensor:
  axis = tensor.dims.index(dim)
  return _unary(tensor, th.softmax, axis, dtype)
