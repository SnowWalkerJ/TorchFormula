import abc
from typing import Any, Dict, Optional, Tuple
import torch as th
import torch.nn as nn
from named_tensor import NamedTensor


class TorchFormula(abc.ABC):
  variable: dict
  @abc.abstractmethod
  def get_dependency(self) -> Dict[str, TorchFormula]:
    raise NotImplementedError

  @abc.abstractmethod
  def compile_offline(self, deps: Dict[str, NamedTensor]) -> Tuple[Optional[nn.Module], Any]:
    raise NotImplementedError

  @abc.abstractmethod
  def compute_offline(self, deps: Dict[str, th.Tensor], module: Optional[nn.Module]):
    raise NotImplementedError

  @abc.abstractmethod
  def initialize(self, deps: Dict[str, NamedTensor], module: Optional[nn.Module]):
    raise NotImplementedError

  @abc.abstractmethod
  def compute_online(self, deps: Dict[str, th.Tensor], module: Optional[nn.Module]):
    raise NotImplementedError
