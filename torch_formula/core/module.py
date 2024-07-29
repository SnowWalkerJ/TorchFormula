import abc
import torch.nn as nn
from .formula import TorchFormula


class TFModule(abc.ABC):
  @abc.abstractmethod
  def __call__(self, *args, **kwargs) -> TorchFormula:
    raise NotImplementedError
