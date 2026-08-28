"""测试Gram矩阵"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from functional_solver.core.data_container import MultiDimData
from functional_solver.core.basis_container import BasisSet
from functional_solver.basis.factory import BasisFactory
from functional_solver.inner_product.base import InnerProduct

def test_gram_matrix_1d():
    """测试1D Gram矩阵计算"""
    # 创建基函数集合
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=2))
    
    # 创建数据
    x = np.linspace(0, 1, 10)
    data = MultiDimData({0: x})
    
    # 创建内积（离散内积）
    inner_product = InnerProduct(is_continuous=False)
    
    # 计算基函数矩阵
    Phi = basis_set.evaluate_all(data)
    
    # 计算Gram矩阵
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 验证Gram矩阵的性质
    assert G.shape == (3, 3)  # 3个基函数
    assert np.allclose(G, G.T)  # 对称
    
    # 验证Gram矩阵的值
    # 对于离散内积，G[i,j] = Σ φ_i(x_k) φ_j(x_k)
    expected = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            expected[i, j] = np.sum(x**i * x**j)
    
    assert np.allclose(G, expected, rtol=1e-10)

def test_gram_matrix_continuous():
    """测试连续内积的Gram矩阵"""
    # 创建基函数集合
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    
    # 创建数据
    x = np.linspace(0, 1, 100)  # 更多点以获得更好的积分精度
    data = MultiDimData({0: x})
    
    # 创建连续内积
    inner_product = InnerProduct(is_continuous=True)
    
    # 计算基函数矩阵
    Phi = basis_set.evaluate_all(data)
    
    # 计算Gram矩阵
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 验证Gram矩阵的性质
    assert G.shape == (2, 2)
    assert np.allclose(G, G.T)  # 对称
    
    # 验证Gram矩阵的值（近似积分）
    # ∫₀¹ 1 dx = 1
    # ∫₀¹ x dx = 0.5
    # ∫₀¹ x² dx = 1/3
    expected = np.array([[1.0, 0.5], [0.5, 1.0/3]])
    
    assert np.allclose(G, expected, rtol=1e-2)  # 允许2%的误差

def test_weighted_inner_product():
    """测试加权内积"""
    # 创建基函数集合
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=1))
    
    # 创建数据
    x = np.linspace(0, 1, 10)
    data = MultiDimData({0: x})
    
    # 创建加权内积（权重函数 w(x) = x）
    def weight_func(x):
        return x
    
    inner_product = InnerProduct(weight_func=weight_func, is_continuous=False)
    
    # 计算基函数矩阵
    Phi = basis_set.evaluate_all(data)
    
    # 计算Gram矩阵
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 验证Gram矩阵的值
    # 对于加权离散内积，G[i,j] = Σ w(x_k) φ_i(x_k) φ_j(x_k)
    expected = np.zeros((2, 2))
    w = weight_func(x)
    expected[0, 0] = np.sum(w * x**0 * x**0)  # Σ w(x) * 1 * 1
    expected[0, 1] = np.sum(w * x**0 * x**1)  # Σ w(x) * 1 * x
    expected[1, 0] = expected[0, 1]  # 对称
    expected[1, 1] = np.sum(w * x**1 * x**1)  # Σ w(x) * x * x
    
    assert np.allclose(G, expected, rtol=1e-10)

def test_gram_matrix_orthogonal_basis():
    """测试正交基函数的Gram矩阵"""
    # 使用勒让德多项式（在[-1,1]上正交）
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.legendre(dim=0, order=0))
    basis_set.add_basis(BasisFactory.legendre(dim=0, order=1))
    basis_set.add_basis(BasisFactory.legendre(dim=0, order=2))
    
    # 创建数据（在[-1,1]区间）
    x = np.linspace(-1, 1, 100)
    data = MultiDimData({0: x})
    
    # 创建内积（离散近似）
    inner_product = InnerProduct(is_continuous=False)
    
    # 计算基函数矩阵
    Phi = basis_set.evaluate_all(data)
    
    # 计算Gram矩阵
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 验证Gram矩阵近似对角（勒让德多项式在离散点上近似正交）
    # 提取对角线
    diag = np.diag(G)
    
    # 非对角线元素应该相对较小
    G_off_diag = G.copy()
    np.fill_diagonal(G_off_diag, 0)
    off_diag_norm = np.linalg.norm(G_off_diag, 'fro')
    
    # 非对角线元素的范数应该远小于对角线元素的范数
    assert off_diag_norm < 0.1 * np.linalg.norm(diag)

def test_gram_matrix_singularity():
    """测试Gram矩阵的奇异性检测"""
    from functional_solver.inner_product.regularization import Regularization
    
    # 创建线性相关的基函数
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))  # 常数1
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))  # 另一个常数1（线性相关）
    
    # 创建数据
    x = np.linspace(0, 1, 10)
    data = MultiDimData({0: x})
    
    # 创建内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 计算基函数矩阵
    Phi = basis_set.evaluate_all(data)
    
    # 计算Gram矩阵
    G = inner_product.compute_gram_matrix(Phi, data)
    
    # 检测奇异性
    singularity_info = Regularization.check_singularity(G, threshold=1e-10)
    
    # Gram矩阵应该是奇异的（因为基函数线性相关）
    assert singularity_info["is_singular"] == True
    assert singularity_info["rank"] < G.shape[0]

def test_gram_matrix_regularization():
    """测试Gram矩阵的正则化"""
    from functional_solver.inner_product.regularization import Regularization
    
    # 创建一个接近奇异的Gram矩阵
    G = np.array([[1.0, 0.999999], [0.999999, 1.0]])
    
    # 应用Tikhonov正则化
    G_reg = Regularization.tikhonov(G, alpha=1e-6)
    
    # 验证正则化后的矩阵
    assert G_reg.shape == G.shape
    assert np.allclose(G_reg, G_reg.T)  # 仍然对称
    
    # 对角线增加了alpha
    assert np.allclose(np.diag(G_reg), np.diag(G) + 1e-6)
    
    # 条件数应该改善
    cond_original = np.linalg.cond(G)
    cond_regularized = np.linalg.cond(G_reg)
    assert cond_regularized < cond_original

def test_gram_solver():
    """测试Gram解算器"""
    from functional_solver.solver.gram_solver import GramSolver
    
    # 创建基函数集合
    basis_set = BasisSet()
    for order in range(3):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    
    # 创建内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建Gram解算器
    solver = GramSolver(basis_set, inner_product)
    
    # 创建数据
    x = np.linspace(0, 1, 20)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))  # 添加噪声
    
    data = MultiDimData({0: x})
    
    # 加载数据
    solver.load_data(data, y)
    
    # 计算Gram矩阵
    G = solver.compute_gram_matrix()
    assert G.shape == (3, 3)
    
    # 计算右端项
    b = solver.compute_rhs()
    assert b.shape == (3,)
    
    # 求解
    coefficients = solver.solve()
    assert coefficients.shape == (3,)
    
    # 预测
    y_pred = solver.predict(data)
    assert y_pred.shape == (20,)
    
    # 计算条件数
    cond = solver.get_condition_number()
    assert cond > 0
    
    # 计算奇异值
    singular_values = solver.get_singular_values()
    assert len(singular_values) == 3

if __name__ == "__main__":
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
    
    print("\n所有Gram矩阵测试通过！")
