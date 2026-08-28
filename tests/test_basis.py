"""Tests for the basis function module."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory

def test_polynomial_basis():
    """Test the polynomial basis."""
    basis = BasisFactory.polynomial(dim=0, order=3)
    assert basis.name == "Polynomial_order3"
    assert basis.dim == 0
    
    # 在 x = 2.0 处计算基函数的值：多项式基的定义是 x^order，这里 order=3，即 2³ = 8
    x = 2.0
    val = basis.func(x, order=3)
    assert val == 8.0

def test_basis_set_evaluation():
    """Test evaluation of a basis set."""
    basis_set = BasisSet()
    
    # 往基集合里添加两个基函数：0 阶常数基（恒等于 1）和 1 阶线性基（x），都作用于维度 0
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    
    # 构造测试数据：维度 0 上有三个采样点 x = 1, 2, 3
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0])
    })
    
    # 在所有采样点上同时求值全部基函数，得到设计矩阵 Phi（行 = 采样点，列 = 基函数）
    Phi = basis_set.evaluate_all(data)
    
    # 验证设计矩阵：第 1 列全是 1（常数基的取值），第 2 列就是 x 本身（线性基的取值）
    expected = np.array([[1, 1], [1, 2], [1, 3]])
    assert np.allclose(Phi, expected)

def test_fourier_basis():
    """Test the Fourier basis."""
    bases = BasisFactory.fourier(dim=0, freq=1, L=2*np.pi)
    
    # 频率 f = 1 时应生成两个基函数：余弦基与正弦基（傅里叶基总是成对出现）
    assert len(bases) == 2
    
    # 先测试余弦基：在 x = π/2 处求值
    cos_basis = bases[0]
    x = np.pi/2
    val = cos_basis.func(x, freq=1, L=2*np.pi)
    # 余弦基的公式为 cos(2πf·x/L)，代入 f=1、L=2π 后化简为 cos(x)，与 numpy 直接计算的结果对比
    expected = np.cos(2*np.pi*1*x/(2*np.pi))
    assert np.allclose(val, expected)
    
    # 再测试正弦基：同理验证 sin(2πf·x/L) 的取值
    sin_basis = bases[1]
    val = sin_basis.func(x, freq=1, L=2*np.pi)
    expected = np.sin(2*np.pi*1*x/(2*np.pi))
    assert np.allclose(val, expected)

def test_legendre_basis():
    """Test the Legendre polynomial basis."""
    basis = BasisFactory.legendre(dim=0, order=2)
    assert basis.name == "Legendre_order2"
    assert basis.dim == 0
    
    # 在 x = 0.5 处计算二阶勒让德基函数的值
    x = 0.5
    val = basis.func(x, order=2)
    # 二阶勒让德多项式的解析公式：P₂(x) = (3x² - 1)/2，据此直接写出期望值
    expected = (3*x**2 - 1)/2
    assert np.allclose(val, expected)

def test_custom_basis():
    """Test a custom basis function."""
    def custom_func(x, scale=1.0):
        return np.exp(-scale * x**2)
    
    basis = BasisFactory.custom(dim=0, name="Gaussian", func=custom_func, params={"scale": 0.5})
    assert basis.name == "Gaussian"
    assert basis.dim == 0
    assert "scale" in basis.params
    
    # 在 x = 1.0、scale = 0.5 的参数下调用自定义基函数
    x = 1.0
    val = basis.func(x, scale=0.5)
    # 期望值 = exp(-scale·x²) = exp(-0.5·1²)，与 custom_func 的公式完全一致
    expected = np.exp(-0.5 * x**2)
    assert np.allclose(val, expected)

def test_basis_set_dimension_organization():
    """Test dimension organization in a basis set."""
    basis_set = BasisSet()
    
    # 分别给维度 0 和维度 1 各添加常数基与线性基，用来测试基集合按维度分组管理
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=1))
    
    # 检查按维度组织的结构：by_dim 字典以维度号为键，值为该维度下的基函数列表
    assert len(basis_set) == 4
    assert 0 in basis_set.by_dim
    assert 1 in basis_set.by_dim
    assert len(basis_set.by_dim[0]) == 2
    assert len(basis_set.by_dim[1]) == 2
    
    # 检查按维度取回基函数的接口：get_bases_for_dim(0) 应返回维度 0 上的两个基函数
    dim0_bases = basis_set.get_bases_for_dim(0)
    assert len(dim0_bases) == 2
    for basis in dim0_bases:
        assert basis.dim == 0

def test_basis_set_evaluate_at_point():
    """Test basis set evaluation at a single point."""
    basis_set = BasisSet()
    
    # 同样准备维度 0、1 上的常数基与线性基
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=1, order=1))
    
    # 在单个点 (x=2.0, y=3.0) 上同时求值所有基函数
    point = {0: 2.0, 1: 3.0}
    values = basis_set.evaluate_at_point(point)
    
    # 期望值依次为 [1, 2, 1, 3]：常数基 1、线性基 x=2、常数基 1、线性基 y=3
    expected = np.array([1.0, 2.0, 1.0, 3.0])
    assert np.allclose(values, expected)

def test_rbf_basis():
    """Test the radial basis function (RBF)."""
    from geomfit.basis.rbf import RBFBasis
    
    # 构造高斯径向基函数（RBF）：中心 center=0.0，宽度参数 sigma=1.0
    rbf = RBFBasis.gaussian(dim=0, center=0.0, sigma=1.0)
    assert rbf.name.startswith("Gaussian_RBF")
    assert rbf.dim == 0
    
    # 在 x = 1.0 处计算该 RBF 的值
    x = 1.0
    val = rbf.func(x, center=0.0, sigma=1.0)
    # 高斯 RBF 公式：exp(-(x - center)² / (2σ²))，代入 center=0、σ=1 即得到下面的期望表达式
    expected = np.exp(-(x - 0.0)**2 / (2 * 1.0**2))
    assert np.allclose(val, expected)

def test_wavelet_basis():
    """Test the wavelet basis function."""
    from geomfit.basis.wavelet import WaveletBasis
    
    # 构造墨西哥帽小波基函数：中心 center=0.0，尺度 scale=1.0
    wavelet = WaveletBasis.mexican_hat(dim=0, center=0.0, scale=1.0)
    assert wavelet.name.startswith("MexicanHat")
    assert wavelet.dim == 0
    
    # 在 x = 0.5 处调用该小波基函数，验证其可调用且返回数值
    x = 0.5
    val = wavelet.func(x, center=0.0, scale=1.0)
    # 墨西哥帽小波公式：ψ(x) = (1 - x²)·exp(-x²/2)；输入需先按 (x - center)/scale 归一化
    x_norm = (x - 0.0) / 1.0
    expected = (1 - x_norm**2) * np.exp(-x_norm**2 / 2)
    assert np.allclose(val, expected)

if __name__ == "__main__":
    # 直接运行本脚本时，按顺序执行全部测试用例并打印通过信息
    test_polynomial_basis()
    test_basis_set_evaluation()
    test_fourier_basis()
    test_legendre_basis()
    test_custom_basis()
    test_basis_set_dimension_organization()
    test_basis_set_evaluate_at_point()
    test_rbf_basis()
    test_wavelet_basis()
    print("All tests passed!")
