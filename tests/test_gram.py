"""Tests for the Gram matrix."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory
from geomfit.inner_product.base import InnerProduct

def test_gram_matrix_1d():
    """Test Gram matrix computation in 1D."""
    # 构造基集合：0、1、2 阶多项式基 {1, x, x²}
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=2))
    
    # 生成数据：在区间 [0, 1] 上均匀取 10 个采样点
    x = np.linspace(0, 1, 10)
    data = MultiDimData({0: x})
    
    # 创建离散内积（用采样点上的求和来近似积分）
    inner_product = InnerProduct(is_continuous=False)
    
    # 计算设计矩阵 Phi：行 = 采样点，列 = 基函数在该点上的取值
    Phi = basis_set.evaluate_all(data)
    
    # 计算 Gram 矩阵 G = ΦᵀΦ，即各基函数两两之间的内积
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 验证 Gram 矩阵的基本性质
    assert G.shape == (3, 3)  # 3 个基函数 → 3×3 矩阵
    assert np.allclose(G, G.T)  # 内积具有对称性，Gram 矩阵必为对称阵
    
    # 逐元素验证 Gram 矩阵的数值
    # 离散内积的定义：G[i,j] = Σₖ φ_i(xₖ)·φ_j(xₖ)，即对所有采样点求和
    expected = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            # 多项式基 φ_i(x) = x^i，因此 φ_i(xₖ)·φ_j(xₖ) = xₖ^i · xₖ^j = xₖ^(i+j)
            expected[i, j] = np.sum(x**i * x**j)
    
    assert np.allclose(G, expected, rtol=1e-10)

def test_gram_matrix_continuous():
    """Test the Gram matrix with a continuous inner product."""
    # 构造基集合：0、1 阶多项式基 {1, x}
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    
    # 生成数据：采样点越多（100 个），离散求和越接近精确积分
    x = np.linspace(0, 1, 100)  # 更多采样点，提高数值积分精度
    data = MultiDimData({0: x})
    
    # 创建连续内积（通过密集采样点做数值积分来近似 ∫₀¹ f(x)·g(x) dx）
    inner_product = InnerProduct(is_continuous=True)
    
    # 计算设计矩阵
    Phi = basis_set.evaluate_all(data)
    
    # 计算 Gram 矩阵
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 验证 Gram 矩阵的基本性质
    assert G.shape == (2, 2)
    assert np.allclose(G, G.T)  # 内积对称，Gram 矩阵必为对称阵
    
    # 验证 Gram 矩阵元素（即各个积分的数值近似）
    # 解析积分结果：∫₀¹ 1 dx = 1
    # ∫₀¹ x dx = 0.5
    # ∫₀¹ x² dx = 1/3
    expected = np.array([[1.0, 0.5], [0.5, 1.0/3]])
    
    assert np.allclose(G, expected, rtol=1e-2)  # 数值积分存在离散化误差，允许 2% 的相对误差

def test_weighted_inner_product():
    """Test the weighted inner product."""
    # 构造基集合：0、1 阶多项式基
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    
    # 生成数据：10 个均匀采样点
    x = np.linspace(0, 1, 10)
    data = MultiDimData({0: x})
    
    # 创建带权内积：权函数 w(x) = x（越靠右的采样点权重越大）
    def weight_func(x):
        return x
    
    inner_product = InnerProduct(weight_func=weight_func, is_continuous=False)
    
    # 计算设计矩阵
    Phi = basis_set.evaluate_all(data)
    
    # 计算带权 Gram 矩阵
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 逐元素验证带权 Gram 矩阵
    # 带权离散内积的定义：G[i,j] = Σₖ w(xₖ)·φ_i(xₖ)·φ_j(xₖ)
    expected = np.zeros((2, 2))
    w = weight_func(x)
    expected[0, 0] = np.sum(w * x**0 * x**0)  # G[0,0] = Σ w(x)·1·1
    expected[0, 1] = np.sum(w * x**0 * x**1)  # G[0,1] = Σ w(x)·1·x
    expected[1, 0] = expected[0, 1]  # 由对称性 G[1,0] = G[0,1]
    expected[1, 1] = np.sum(w * x**1 * x**1)  # G[1,1] = Σ w(x)·x·x
    
    assert np.allclose(G, expected, rtol=1e-10)

def test_gram_matrix_orthogonal_basis():
    """Test the Gram matrix of orthogonal basis functions."""
    # 使用勒让德多项式基：它们在区间 [-1, 1] 上两两正交
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.legendre(dim=0, order=0))
    basis_set.add_basis(BasisFactory.legendre(dim=0, order=1))
    basis_set.add_basis(BasisFactory.legendre(dim=0, order=2))
    
    # 在勒让德多项式的正交区间 [-1, 1] 上采样 100 个点
    x = np.linspace(-1, 1, 100)
    data = MultiDimData({0: x})
    
    # 创建内积（用离散求和近似连续内积）
    inner_product = InnerProduct(is_continuous=False)
    
    # 计算设计矩阵
    Phi = basis_set.evaluate_all(data)
    
    # 计算 Gram 矩阵
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 勒让德多项式在离散采样点上近似正交，因此 Gram 矩阵应接近对角阵
    # （即非对角元素应远小于对角元素）
    # 取出对角元素：各基函数与自身的内积（相当于范数的平方）
    diag = np.diag(G)
    
    # 把对角元素清零，只保留非对角部分，衡量"非正交成分"的总量
    G_off_diag = G.copy()
    np.fill_diagonal(G_off_diag, 0)
    off_diag_norm = np.linalg.norm(G_off_diag, 'fro')
    
    # 非对角部分的 Frobenius 范数应远小于对角部分的范数，从而验证近似正交性
    assert off_diag_norm < 0.1 * np.linalg.norm(diag)

def test_gram_matrix_singularity():
    """Test singularity detection for the Gram matrix."""
    from geomfit.inner_product.regularization import Regularization
    
    # 构造线性相关的基函数：两个完全相同的常数基 1
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))  # 常数基 1
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))  # 再添加一个相同的常数基，二者线性相关
    
    # 生成数据：10 个采样点
    x = np.linspace(0, 1, 10)
    data = MultiDimData({0: x})
    
    # 创建离散内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 计算设计矩阵
    Phi = basis_set.evaluate_all(data)
    
    # 计算 Gram 矩阵
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 调用正则化模块检测奇异性：检查矩阵是否不满秩
    singularity_info = Regularization.check_singularity(G, threshold=1e-10)
    
    # 由于基函数线性相关，Gram 矩阵应被判定为奇异（秩 < 维数）
    assert singularity_info["is_singular"] == True
    assert singularity_info["rank"] < G.shape[0]

def test_gram_matrix_regularization():
    """Test Gram matrix regularization."""
    from geomfit.inner_product.regularization import Regularization
    
    # 构造一个接近奇异的 Gram 矩阵（两行几乎相同，条件数极大）
    G = np.array([[1.0, 0.999999], [0.999999, 1.0]])
    
    # 施加 Tikhonov（岭）正则化：G + α·I，这里 α = 1e-6
    G_reg = Regularization.tikhonov(G, alpha=1e-6)
    
    # 验证正则化后的矩阵
    assert G_reg.shape == G.shape
    assert np.allclose(G_reg, G_reg.T)  # 加上对角项不会破坏对称性
    
    # 正则化等价于把对角元素各加上 α
    assert np.allclose(np.diag(G_reg), np.diag(G) + 1e-6)
    
    # 条件数（最大奇异值 / 最小奇异值）应因正则化而下降，数值稳定性变好
    cond_original = np.linalg.cond(G)
    cond_regularized = np.linalg.cond(G_reg)
    assert cond_regularized < cond_original

def test_gram_solver():
    """Test the GramSolver."""
    from geomfit.solver.gram_solver import GramSolver
    
    # 构造基集合：0、1、2 阶多项式基
    basis_set = BasisSet()
    for order in range(3):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    
    # 创建离散内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建 Gram 求解器：把基集合与内积组合在一起
    solver = GramSolver(basis_set, inner_product)
    
    # 生成测试数据：在 [0, 1] 上取 20 个点，目标值为正弦函数叠加噪声
    x = np.linspace(0, 1, 20)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))  # 叠加高斯噪声，模拟带误差的观测值
    
    data = MultiDimData({0: x})
    
    # 把数据与目标值载入求解器
    solver.load_data(data, y)
    
    # 计算 Gram 矩阵
    G = solver.compute_gram_matrix()
    assert G.shape == (3, 3)
    
    # 计算右端项 b：各基函数与目标值的内积
    b = solver.compute_rhs()
    assert b.shape == (3,)
    
    # 求解线性方程组 G·c = b，得到基函数系数
    coefficients = solver.solve()
    assert coefficients.shape == (3,)
    
    # 用学到的系数在数据点上做预测
    y_pred = solver.predict(data)
    assert y_pred.shape == (20,)
    
    # 计算 Gram 矩阵的条件数，评估数值稳定性
    cond = solver.get_condition_number()
    assert cond > 0
    
    # 计算奇异值（数量应与基函数个数一致，即 3 个）
    singular_values = solver.get_singular_values()
    assert len(singular_values) == 3

if __name__ == "__main__":
    # 直接运行本脚本时，顺序执行全部测试用例并打印通过信息
    test_gram_matrix_1d()
    print("✓ test_gram_matrix_1d passed")
    
    test_gram_matrix_continuous()
    print("✓ test_gram_matrix_continuous passed")
    
    test_weighted_inner_product()
    print("✓ test_weighted_inner_product passed")
    
    test_gram_matrix_orthogonal_basis()
    print("✓ test_gram_matrix_orthogonal_basis passed")
    
    test_gram_matrix_singularity()
    print("✓ test_gram_matrix_singularity passed")
    
    test_gram_matrix_regularization()
    print("✓ test_gram_matrix_regularization passed")
    
    test_gram_solver()
    print("✓ test_gram_solver passed")
    
    print("\nAll Gram matrix tests passed!")
