import torch
import numpy as np
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete import PTDiscreteFactor

def test_sum():
    print("Testing sum PT factors..")
    phi1_value = torch.randn(12) * 10
    phi2_value = torch.randn(8) * 10
    phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], phi1_value.numpy())
    phi2 = DiscreteFactor(['x3', 'x4', 'x1'], [2, 2, 2], phi2_value.numpy())
    phi1.sum(phi2, inplace=True)
    phi1.variables
    
    nphi1 = PTDiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], phi1_value)
    nphi2 = PTDiscreteFactor(['x3', 'x4', 'x1'], [2, 2, 2], phi2_value)
    nphi1.sum(nphi2, inplace=True)
    nphi1.variables

    assert phi1.variables == nphi1.variables
    print(phi1.values - nphi1.values.numpy())
    assert np.absolute(phi1.values - nphi1.values.numpy()).sum() < 1e-5

def test_product():
    print("Testing product PT factors..")
    phi1_value = torch.randn(12) * 10
    phi2_value = torch.randn(8) * 10
    phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], phi1_value.numpy())
    phi2 = DiscreteFactor(['x3', 'x4', 'x1'], [2, 2, 2], phi2_value.numpy())
    phi1.product(phi2, inplace=True)
    phi1.variables
    
    nphi1 = PTDiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], phi1_value)
    nphi2 = PTDiscreteFactor(['x3', 'x4', 'x1'], [2, 2, 2], phi2_value)
    nphi1.product(nphi2, inplace=True)
    nphi1.variables

    assert phi1.variables == nphi1.variables
    print(phi1.values - nphi1.values.numpy())
    assert np.absolute(phi1.values - nphi1.values.numpy()).sum() < 5e-5, "error {}".format(np.absolute(phi1.values - nphi1.values.numpy()).sum())


def test_divide():
    print("Testing devide PT factors..")
    phi1_value = torch.randn(12) * 10
    phi2_value = torch.randn(4) * 10
    phi1 = DiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], phi1_value.numpy())
    phi2 = DiscreteFactor(['x3', 'x1'], [2, 2], phi2_value.numpy())
    phi1.divide(phi2, inplace=True)
    phi1.variables
    
    nphi1 = PTDiscreteFactor(['x1', 'x2', 'x3'], [2, 3, 2], phi1_value)
    nphi2 = PTDiscreteFactor(['x3', 'x1'], [2, 2], phi2_value)
    nphi1.divide(nphi2, inplace=True)
    nphi1.variables

    assert phi1.variables == nphi1.variables
    print(phi1.values - nphi1.values.numpy())
    assert np.absolute(phi1.values - nphi1.values.numpy()).sum() < 1e-5



if __name__ == "__main__":
    test_sum()
    test_product()
    test_divide()
