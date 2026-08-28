"""测试解算器"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from functional_solver.core.data_container import MultiDimData
from functional_solver.core.basis_container import BasisSet
from functional_solver.basis.factory import BasisFactory
from functional_solver.inner_product.base import InnerProduct
from functional_solver.solver.functional_solver import FunctionalSolver
from functional_solver.solver.gram_solver import GramSolver
from functional_solver.solver.kernel_solver import KernelSolver
from functional_solver.kernel.rbf import RBFKernel

def test_functional_solver_basis_method():
    """测试FunctionalSolver的基函数方法"""
    # 创建解算器
    solver = FunctionalSolver()
    
    # 创建基函数集合
    basis_set = BasisSet()
    for order in range(5):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    
    # 设置基函数
    solver.set_basis(basis_set)
    
    # 创建内积
    inner_product = InnerProduct(is_continuous=False)
    solver.set_inner_product(inner_product)
    
    # 创建数据
    x = np.linspace(0, 1, 50)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))  # 添加噪声
    
    data = MultiDimData({0: x})
    
    # 加载数据
    solver.load_data(data, y)
    
    # 求解（无正则化）
    coefficients = solver.solve()
    assert coefficients.shape == (5,)
    
    # 预测
    y_pred = solver.predict(data)
    assert y_pred.shape == (50,)
    
    # 计算均方误差
    mse = np.mean((y_pred - y)**2)
    assert mse < 0.05  # 误差应该较小
    
    # 获取调试信息
    debug_info = solver.get_debug_info()
    assert "method" in debug_info
    assert debug_info["method"] == "basis"

def test_functional_solver_kernel_method():
    """测试FunctionalSolver的核方法"""
    # 创建解算器
    solver = FunctionalSolver()
    
    # 创建核函数
    kernel = RBFKernel(sigma=1.0)
    solver.set_kernel(kernel)
    
    # 创建数据
    x = np.linspace(0, 1, 30)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))
    
    data = MultiDimData({0: x})
    
    # 加载数据
    solver.load_data(data, y)
    
    # 求解
    coefficients = solver.solve()
    assert coefficients.shape == (30,)  # 核方法系数数量等于数据点数量
    
    # 预测
    y_pred = solver.predict(data)
    assert y_pred.shape == (30,)
    
    # 计算均方误差
    mse = np.mean((y_pred - y)**2)
    assert mse < 0.1  # 核方法应该能很好拟合
    
    # 获取调试信息
    debug_info = solver.get_debug_info()
    assert "method" in debug_info
    assert debug_info["method"] == "kernel"

def test_functional_solver_regularization():
    """测试FunctionalSolver的正则化"""
    # 创建解算器
    solver = FunctionalSolver()
    
    # 创建基函数集合（使用高阶多项式，容易过拟合）
    basis_set = BasisSet()
    for order in range(15):  # 高阶多项式
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    
    solver.set_basis(basis_set)
    
    # 创建内积
    inner_product = InnerProduct(is_continuous=False)
    solver.set_inner_product(inner_product)
    
    # 创建数据
    x = np.linspace(0, 1, 20)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.2 * np.random.randn(len(x))  # 较大噪声
    
    data = MultiDimData({0: x})
    solver.load_data(data, y)
    
    # 无正则化求解
    coeff_no_reg = solver.solve()
    
    # 带正则化求解
    regularization = {"method": "tikhonov", "alpha": 1e-3}
    coeff_with_reg = solver.solve(regularization=regularization)
    
    # 正则化后的系数范数应该更小
    norm_no_reg = np.linalg.norm(coeff_no_reg)
    norm_with_reg = np.linalg.norm(coeff_with_reg)
    assert norm_with_reg < norm_no_reg

def test_functional_solver_reset():
    """测试FunctionalSolver的重置功能"""
    solver = FunctionalSolver()
    
    # 设置一些状态
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    solver.set_basis(basis_set)
    
    inner_product = InnerProduct()
    solver.set_inner_product(inner_product)
    
    x = np.linspace(0, 1, 10)
    y = np.sin(x)
    data = MultiDimData({0: x})
    solver.load_data(data, y)
    
    # 求解
    solver.solve()
    
    # 重置
    solver.reset()
    
    # 验证状态已重置
    assert solver.basis_set is None
    assert solver.inner_product is None
    assert solver.data is None
    assert solver.target is None
    assert solver.coefficients is None
    assert solver.debug_info == {}

def test_gram_solver_basic():
    """测试GramSolver基本功能"""
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
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    
    # 加载数据
    solver.load_data(data, y)
    
    # 计算Gram矩阵
    G = solver.compute_gram_matrix()
    assert G.shape == (3, 3)
    assert np.allclose(G, G.T)  # 对称
    
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
    sv = solver.get_singular_values()
    assert len(sv) == 3
    assert np.all(sv >= 0)

def test_kernel_solver_basic():
    """测试KernelSolver基本功能"""
    # 创建核函数
    kernel = RBFKernel(sigma=1.0)
    
    # 创建核解算器
    solver = KernelSolver(kernel)
    
    # 创建数据
    x = np.linspace(0, 1, 20)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    
    # 加载数据
    solver.load_data(data, y)
    
    # 计算核矩阵
    K = solver.compute_kernel_matrix()
    assert K.shape == (20, 20)
    assert np.allclose(K, K.T)  # 对称
    
    # 求解
    coefficients = solver.solve()
    assert coefficients.shape == (20,)
    
    # 预测
    y_pred = solver.predict(data)
    assert y_pred.shape == (20,)
    
    # 高效预测
    y_pred_eff = solver.predict_efficient(data)
    assert y_pred_eff.shape == (20,)
    assert np.allclose(y_pred, y_pred_eff, rtol=1e-10)
    
    # 计算条件数
    cond = solver.get_condition_number()
    assert cond > 0
    
    # 计算特征值
    eigenvalues = solver.get_eigenvalues()
    assert len(eigenvalues) == 20
    assert np.all(eigenvalues >= 0)  # 核矩阵半正定
    
    # 计算函数范数
    norm = solver.compute_function_norm()
    assert norm >= 0

def test_kernel_solver_regularization():
    """测试KernelSolver的正则化"""
    # 创建核函数
    kernel = RBFKernel(sigma=0.5)
    
    # 创建核解算器
    solver = KernelSolver(kernel)
    
    # 创建数据
    x = np.linspace(0, 1, 15)
    y = np.sin(2 * np.pi * x) + 0.2 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    solver.load_data(data, y)
    
    # 无正则化求解
    coeff_no_reg = solver.solve()
    
    # 带正则化求解
    regularization = {"alpha": 1e-3}
    coeff_with_reg = solver.solve(regularization=regularization)
    
    # 正则化后的系数范数应该更小
    norm_no_reg = np.linalg.norm(coeff_no_reg)
    norm_with_reg = np.linalg.norm(coeff_with_reg)
    assert norm_with_reg < norm_no_reg

def test_solver_comparison():
    """测试不同解算器的比较"""
    # 创建数据
    x = np.linspace(0, 1, 30)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    
    # 测试1: FunctionalSolver with basis
    solver1 = FunctionalSolver()
    basis_set = BasisSet()
    for order in range(5):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    solver1.set_basis(basis_set)
    solver1.set_inner_product(InnerProduct(is_continuous=False))
    solver1.load_data(data, y)
    coeff1 = solver1.solve()
    y_pred1 = solver1.predict(data)
    mse1 = np.mean((y_pred1 - y)**2)
    
    # 测试2: FunctionalSolver with kernel
    solver2 = FunctionalSolver()
    kernel = RBFKernel(sigma=1.0)
    solver2.set_kernel(kernel)
    solver2.load_data(data, y)
    coeff2 = solver2.solve()
    y_pred2 = solver2.predict(data)
    mse2 = np.mean((y_pred2 - y)**2)
    
    # 测试3: GramSolver
    solver3 = GramSolver(basis_set, InnerProduct(is_continuous=False))
    solver3.load_data(data, y)
    coeff3 = solver3.solve()
    y_pred3 = solver3.predict(data)
    mse3 = np.mean((y_pred3 - y)**2)
    
    # 测试4: KernelSolver
    solver4 = KernelSolver(kernel)
    solver4.load_data(data, y)
    coeff4 = solver4.solve()
    y_pred4 = solver4.predict(data)
    mse4 = np.mean((y_pred4 - y)**2)
    
    # 验证所有解算器都能得到合理的结果
    assert mse1 < 0.05
    assert mse2 < 0.05
    assert mse3 < 0.05
    assert mse4 < 0.05
    
    # FunctionalSolver with basis 和 GramSolver 应该得到相同的结果
    assert np.allclose(coeff1, coeff3, rtol=1e-10)
    assert np.allclose(y_pred1, y_pred3, rtol=1e-10)
    
    # FunctionalSolver with kernel 和 KernelSolver 应该得到相同的结果
    assert np.allclose(coeff2, coeff4, rtol=1e-10)
    assert np.allclose(y_pred2, y_pred4, rtol=1e-10)

def test_solver_with_new_data():
    """测试解算器对新数据的预测"""
    # 训练数据
    x_train = np.linspace(0, 1, 20)
    y_train = np.sin(2 * np.pi * x_train) + 0.1 * np.random.randn(len(x_train))
    train_data = MultiDimData({0: x_train})
    
    # 测试数据
    x_test = np.linspace(0.1, 0.9, 15)
    y_test_true = np.sin(2 * np.pi * x_test)
    test_data = MultiDimData({0: x_test})
    
    # 使用FunctionalSolver with basis
    solver = FunctionalSolver()
    basis_set = BasisSet()
    for order in range(5):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    solver.set_basis(basis_set)
    solver.set_inner_product(InnerProduct(is_continuous=False))
    solver.load_data(train_data, y_train)
    solver.solve()
    
    # 对测试数据进行预测
    y_test_pred = solver.predict(test_data)
    
    # 验证预测形状
    assert y_test_pred.shape == (15,)
    
    # 计算测试误差
    test_mse = np.mean((y_test_pred - y_test_true)**2)
    assert test_mse < 0.2  # 测试误差应该合理

if __name__ == "__main__":
    test_functional_solver_basis_method()
    print("✓ test_functional_solver_basis_method passed")
    
    test_functional_solver_kernel_method()
    print("✓ test_functional_solver_kernel_method passed")
    
    test_functional_solver_regularization()
    print("✓ test_functional_solver_regularization passed")
    
    test_functional_solver_reset()
    print("✓ test_functional_solver_reset passed")
    
    test_gram_solver_basic()
    print("✓ test_gram_solver_basic passed")
    
    test_kernel_solver_basic()
    print("✓ test_kernel_solver_basic passed")
    
    test_kernel_solver_regularization()
    print("✓ test_kernel_solver_regularization passed")
    
    test_solver_comparison()
    print("✓ test_solver_comparison passed")
    
    test_solver_with_new_data()
    print("✓ test_solver_with_new_data passed")
    
    print("\n所有解算器测试通过！")
