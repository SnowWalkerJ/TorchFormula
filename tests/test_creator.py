import unittest
import torch as th
import named_tensor as nt


class CreatorTestCase(unittest.TestCase):
  def test_zeros(self):
    zeros = nt.zeros(th.float32, ["date", "snap"], coords={"date": ["20230101", "20230102"], "snap": [0, 1800, 3600]})
    ones = nt.ones_like(zeros)
    self.assertEqual((ones - zeros).sum().item(), 6)


if __name__ == '__main__':
  unittest.main()
