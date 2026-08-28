"""测试基函数模块"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from functional_solver.core.data_container import MultiDimData
from functional_solver.core.basis_container import BasisSet
from functional_solver.basis.factory import BasisFactory

def test_polynomial_basis():
    """测试多项式基"""
    basis = BasisFactory.polynomial(dim=0, order=3)
    assert basis.name == "Polynomial_order3"
    assert basis.dim == 0
    
    # 测试计算
    x = 2.0
    val = basis.func(x, order=3)
    assert val == 8.0

def test_basis_set_evaluation():
    """测试基函数集合的计算"""
    basis_set = BasisSet()
    
    # 添加几个基函数
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    
    # 创建测试数据
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0])
    })
    
    # 计算
    Phi = basis_set.evaluate_all(data)
    
    # 验证
    expected = np.array([[1, 1], [1, 2], [1, 3]])
    assert np.allclose(Phi, expected)

def test_fourier_basis():
    """测试傅里叶基"""
    bases = BasisFactory.fourier(dim=0, freq=1, L=2*np.pi)
    
    # 频率为1时应该返回两个基函数（cos和sin）
    assert len(bases) == 2
    
    # 测试cos基函数
    cos_basis = bases[0]
    x = np.pi/2
    val = cos_basis.func(x, freq=1, L=2*np.pi)
    expected = np.cos(2*np.pi*1*x/(2*np.pi))
    assert np.allclose(val, expected)
    
    # 测试sin基函数
    sin_basis = bases[1]
    val = sin_basis.func(x, freq=1, L=2*np.pi)
    expected = np.sin(2*np.pi*1*x/(2*np.pi))
    assert np.allclose(val, expected)

def test_legendre_basis():
    """测试勒让德多项式基"""
    basis = BasisFactory.legendre(dim=0, order=2)
    assert basis.name == "Legendre_order2"
    assert basis.dim == 0
    
    # 测试计算
    x = 0.5
    val = basis.func(x, order=2)
    # 二阶勒让德多项式: P2(x) = (3x^2 - 1)/2
    expected = (3*x**2 - 1)/2
    assert np.allclose(val, expected)

def test_custom_basis():
    """测试自定义基函数"""
    def custom_func(x, scale=1.0):
        return np.exp(-scale * x**2)
    
    basis = BasisFactory.custom(dim=0, name="Gaussian", func=custom_func, params={"scale": 0.5})
    assert basis.name == "Gaussian"
    assert basis.dim == 0
    assert "scale" in basis.params
    
    # 测试计算
    x = 1.0
    val = basis.func(x, scale=0.5)
    expected = np.exp(-0.5 * x**2)
    assert np.allclose(val, expected)

def test_basis_set_dimension_organization():
    """测试基函数集合的维度组织"""
    basis_set = BasisSet()
    
    # 添加不同维度的基函数
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=1))
    
    # 检查维度组织
    assert len(basis_set) == 4
    assert 0 in basis_set.by_dim
    assert 1 in basis_set.by_dim
    assert len(basis_set.by_dim[0]) == 2
    assert len(basis_set.by_dim[1]) == 2
    
    # 检查获取特定维度的基函数
    dim0_bases = basis_set.get_bases_for_dim(0)
    assert len(dim0_bases) == 2
    for basis in dim0_bases:
        assert basis.dim == 0

def test_basis_set_evaluate_at_point():
    """测试基函数集合在单点的计算"""
    basis_set = BasisSet()
    
    # 添加基函数
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=1))
    
    # 测试在单点的计算
    point = {0: 2.0, 1: 3.0}
    values = basis_set.evaluate_at_point(point)
    
    # 期望值: [1, 2, 1, 3]
    expected = np.array([1.0, 2.0, 1.0, 3.0])
    assert np.allclose(values, expected)

def test_rbf_basis():
    """测试径向基函数"""
    from functional_solver.basis.rbf import RBFBasis
    
    # 创建高斯RBF
    rbf = RBFBasis.gaussian(dim=0, center=0.0, sigma=1.0)
    assert rbf.name.startswith("Gaussian_RBF")
    assert rbf.dim == 0
    
    # 测试计算
    x = 1.0
    val = rbf.func(x, center=0.0, sigma=1.0)
    expected = np.exp(-(x - 0.0)**2 / (2 * 1.0**2))
    assert np.allclose(val, expected)

def test_wavelet_basis():
    """测试小波基函数"""
    from functional_solver.basis.wavelet import WaveletBasis
    
    # 创建墨西哥帽小波
    wavelet = WaveletBasis.mexican_hat(dim=0, center=0.0, scale=1.0)
    assert wavelet.name.startswith("MexicanHat")
    assert wavelet.dim == 0
    
    # 测试计算（验证函数可调用）
    x = 0.5
    val = wavelet.func(x, center=0.0, scale=1.0)
    # 墨西哥帽小波公式: (1 - x^2) * exp(-x^2/2)
    x_norm = (x - 0.0) / 1.0
    expected = (1 - x_norm**2) * np.exp(-x_norm**2 / 2)
    assert np.allclose(val, expected)

if __name__ == "__main__":
    test_polynomial_basis()
    test_basis_set_evaluation()
    test_fourier_basis()
    test_legendre_basis()
    test_custom_basis()
    test_basis_set_dimension_organization()
    test_basis_set_evaluate_at_point()
    test_rbf_basis()
    test_wavelet_basis()
    print("所有测试通过！")
