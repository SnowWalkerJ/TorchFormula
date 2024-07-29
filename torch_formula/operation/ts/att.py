import torch as th
from named_tensor import NamedTensor
from ...core.formula import TorchFormula
from ...core.module import TFModule


class TsAttentionFormula(TorchFormula):
  def __init__(self, Q, K, V, feature_dim="feature"):
    self.Q = Q
    self.K = K
    self.V = V
    self.feature_dim = feature_dim
    super().__init__()

  def get_dependency(self):
    return {
      "Q": self.Q,
      "K": self.K,
      "V": self.V,
    }

  def compile(self, deps):
    Q = deps["Q"]
    K = deps["K"]
    V = deps["V"]
    if "snap" not in Q.dims:
      raise ValueError("Q must have dim 'snap'")
    if "snap" not in K.dims:
      raise ValueError("K must have dim 'snap'")
    if "snap" not in V.dims:
      raise ValueError("V must have dim 'snap'")
    if self.feature_dim not in Q.dims:
      raise ValueError(f"Q must have dim '{self.feature_dim}'")
    if self.feature_dim not in K.dims:
      raise ValueError(f"K must have dim '{self.feature_dim}'")
    if self.feature_dim not in V.dims:
      raise ValueError(f"V must have dim '{self.feature_dim}'")
    snaps = Q.shape[Q.dims.index('snap')]
    self.variable['mask'] = th.triu(th.ones((snaps, snaps), device=Q.data.device), diagonal=1)
    dims = set(Q.dims) | set(K.dims) | set(V.dims)
    return NamedTensor

  def compute(self, deps, module):
    pass