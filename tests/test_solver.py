"""Tests for the solvers."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData
from geomfit.core.basis_container import BasisSet
from geomfit.basis.factory import BasisFactory
from geomfit.inner_product.base import InnerProduct
from geomfit.solver.functional_solver import FunctionalSolver
from geomfit.solver.gram_solver import GramSolver
from geomfit.solver.kernel_solver import KernelSolver
from geomfit.kernel.rbf import RBFKernel

def test_functional_solver_basis_method():
    """Test the FunctionalSolver basis method."""
    # 创建函数式求解器实例
    solver = FunctionalSolver()
    
    # 构造基集合：0~4 阶多项式基 {1, x, x², x³, x⁴}，共 5 个基函数
    basis_set = BasisSet()
    for order in range(5):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    
    # 把基集合设置给求解器（即选择"基函数展开"这一方法）
    solver.set_basis(basis_set)
    
    # 创建离散内积并设置给求解器
    inner_product = InnerProduct(is_continuous=False)
    solver.set_inner_product(inner_product)
    
    # 生成测试数据：在 [0, 1] 上取 50 个点，目标函数为正弦波并叠加噪声
    x = np.linspace(0, 1, 50)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))  # 叠加高斯噪声，模拟带误差的观测值
    
    data = MultiDimData({0: x})
    
    # 载入数据与目标值
    solver.load_data(data, y)
    
    # 不做正则化，直接求解基函数系数（最小二乘拟合）
    coefficients = solver.solve()
    assert coefficients.shape == (5,)
    
    # 用拟合得到的系数在训练数据上做预测
    y_pred = solver.predict(data)
    assert y_pred.shape == (50,)
    
    # 计算预测值与真实观测之间的均方误差 (MSE)
    mse = np.mean((y_pred - y)**2)
    assert mse < 0.05  # 5 阶多项式足以逼近正弦波，拟合误差应很小
    
    # 查看调试信息，确认求解器记录的方法名称为 "basis"
    debug_info = solver.get_debug_info()
    assert "method" in debug_info
    assert debug_info["method"] == "basis"

def test_functional_solver_kernel_method():
    """Test the FunctionalSolver kernel method."""
    # 创建函数式求解器实例
    solver = FunctionalSolver()
    
    # 创建径向基函数（RBF）核，sigma=1.0 控制核的宽度
    kernel = RBFKernel(sigma=1.0)
    solver.set_kernel(kernel)
    
    # 生成测试数据：30 个带噪正弦采样点
    x = np.linspace(0, 1, 30)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))
    
    data = MultiDimData({0: x})
    
    # 载入数据与目标值
    solver.load_data(data, y)
    
    # 求解：核方法的待求量是每个数据点的权重系数
    coefficients = solver.solve()
    assert coefficients.shape == (30,)  # 核方法的系数个数等于数据点个数（30 个）
    
    # 预测
    y_pred = solver.predict(data)
    assert y_pred.shape == (30,)
    
    # 计算均方误差
    mse = np.mean((y_pred - y)**2)
    assert mse < 0.1  # 核方法插值能力强，拟合误差应较小
    
    # 查看调试信息，确认方法名称为 "kernel"
    debug_info = solver.get_debug_info()
    assert "method" in debug_info
    assert debug_info["method"] == "kernel"

def test_functional_solver_regularization():
    """Test FunctionalSolver regularization."""
    # 创建求解器
    solver = FunctionalSolver()
    
    # 构造高达 14 阶的多项式基（15 个基函数），极易过拟合，用来检验正则化的效果
    basis_set = BasisSet()
    for order in range(15):  # 高阶多项式：阶数越高，拟合曲线越容易剧烈振荡
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    
    solver.set_basis(basis_set)
    
    # 创建离散内积
    inner_product = InnerProduct(is_continuous=False)
    solver.set_inner_product(inner_product)
    
    # 生成测试数据：20 个带噪采样点
    x = np.linspace(0, 1, 20)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.2 * np.random.randn(len(x))  # 噪声更大（0.2），进一步加剧过拟合
    
    data = MultiDimData({0: x})
    solver.load_data(data, y)
    
    # 先不加正则化解一次，得到"未正则化"的系数
    coeff_no_reg = solver.solve()
    
    # 再带 Tikhonov 正则化（α = 1e-3）求解一次
    regularization = {"method": "tikhonov", "alpha": 1e-3}
    coeff_with_reg = solver.solve(regularization=regularization)
    
    # 正则化会惩罚过大的系数，因此系数向量的 L2 范数（欧氏长度）应当更小
    norm_no_reg = np.linalg.norm(coeff_no_reg)
    norm_with_reg = np.linalg.norm(coeff_with_reg)
    assert norm_with_reg < norm_no_reg

def test_functional_solver_reset():
    """Test the FunctionalSolver reset."""
    solver = FunctionalSolver()
    
    # 先给求解器设置基、内积并载入数据，构造一个有内部状态的求解器
    basis_set = BasisSet()
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=0))
    solver.set_basis(basis_set)
    
    inner_product = InnerProduct()
    solver.set_inner_product(inner_product)
    
    x = np.linspace(0, 1, 10)
    y = np.sin(x)
    data = MultiDimData({0: x})
    solver.load_data(data, y)
    
    # 求解一次，让求解器内部产生系数、调试信息等状态
    solver.solve()
    
    # 调用 reset() 清空全部状态
    solver.reset()
    
    # 逐项验证：基、内积、数据、目标值、系数都应回到 None，调试信息恢复为空字典
    assert solver.basis_set is None
    assert solver.inner_product is None
    assert solver.data is None
    assert solver.target is None
    assert solver.coefficients is None
    assert solver.debug_info == {}

def test_gram_solver_basic():
    """Test basic GramSolver functionality."""
    # 构造基集合：0、1、2 阶多项式基
    basis_set = BasisSet()
    for order in range(3):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    
    # 创建离散内积
    inner_product = InnerProduct(is_continuous=False)
    
    # 创建 Gram 求解器
    solver = GramSolver(basis_set, inner_product)
    
    # 生成测试数据：20 个带噪正弦采样点
    x = np.linspace(0, 1, 20)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    
    # 载入数据与目标值
    solver.load_data(data, y)
    
    # 计算 Gram 矩阵
    G = solver.compute_gram_matrix()
    assert G.shape == (3, 3)
    assert np.allclose(G, G.T)  # Gram 矩阵应是对称的
    
    # 计算右端项 b
    b = solver.compute_rhs()
    assert b.shape == (3,)
    
    # 求解线性方程组
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
    # Gram 矩阵半正定，奇异值应全部非负
    assert np.all(sv >= 0)

def test_kernel_solver_basic():
    """Test basic KernelSolver functionality."""
    # 创建 RBF 核
    kernel = RBFKernel(sigma=1.0)
    
    # 创建核求解器
    solver = KernelSolver(kernel)
    
    # 生成测试数据：20 个带噪正弦采样点
    x = np.linspace(0, 1, 20)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    
    # 载入数据与目标值
    solver.load_data(data, y)
    
    # 计算核矩阵 K：每个元素 K[i,j] = k(x_i, x_j)
    K = solver.compute_kernel_matrix()
    assert K.shape == (20, 20)
    assert np.allclose(K, K.T)  # 径向基（RBF）核是对称的，核矩阵也应是对称的
    
    # 求解：得到每个数据点的权重系数
    coefficients = solver.solve()
    assert coefficients.shape == (20,)
    
    # 普通预测
    y_pred = solver.predict(data)
    assert y_pred.shape == (20,)
    
    # 高效预测：利用核矩阵的结构特性加速计算，结果应与普通预测一致
    y_pred_eff = solver.predict_efficient(data)
    assert y_pred_eff.shape == (20,)
    assert np.allclose(y_pred, y_pred_eff, rtol=1e-10)
    
    # 计算条件数
    cond = solver.get_condition_number()
    assert cond > 0
    
    # 计算核矩阵的特征值
    eigenvalues = solver.get_eigenvalues()
    assert len(eigenvalues) == 20
    assert np.all(eigenvalues >= 0)  # 核矩阵半正定，特征值应非负
    
    # 计算函数在再生核希尔伯特空间中的范数（应非负）
    norm = solver.compute_function_norm()
    assert norm >= 0

def test_kernel_solver_regularization():
    """Test KernelSolver regularization."""
    # 创建 RBF 核（sigma=0.5，宽度更窄）
    kernel = RBFKernel(sigma=0.5)
    
    # 创建核求解器
    solver = KernelSolver(kernel)
    
    # 生成测试数据：15 个带噪正弦采样点
    x = np.linspace(0, 1, 15)
    y = np.sin(2 * np.pi * x) + 0.2 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    solver.load_data(data, y)
    
    # 先不加正则化解一次
    coeff_no_reg = solver.solve()
    
    # 再带岭正则化（α = 1e-3）求解一次
    regularization = {"alpha": 1e-3}
    coeff_with_reg = solver.solve(regularization=regularization)
    
    # 正则化后系数向量的 L2 范数应当更小
    norm_no_reg = np.linalg.norm(coeff_no_reg)
    norm_with_reg = np.linalg.norm(coeff_with_reg)
    assert norm_with_reg < norm_no_reg

def test_solver_comparison():
    """Test comparison across solvers."""
    # 生成共享的测试数据：30 个带噪正弦采样点
    x = np.linspace(0, 1, 30)
    y_true = np.sin(2 * np.pi * x)
    y = y_true + 0.1 * np.random.randn(len(x))
    data = MultiDimData({0: x})
    
    # 求解器 1：函数式求解器的基函数方法（5 阶多项式）
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
    
    # 求解器 2：函数式求解器的核方法（RBF 核）
    solver2 = FunctionalSolver()
    kernel = RBFKernel(sigma=1.0)
    solver2.set_kernel(kernel)
    solver2.load_data(data, y)
    coeff2 = solver2.solve()
    y_pred2 = solver2.predict(data)
    mse2 = np.mean((y_pred2 - y)**2)
    
    # 求解器 3：Gram 求解器（与求解器 1 数学上等价）
    solver3 = GramSolver(basis_set, InnerProduct(is_continuous=False))
    solver3.load_data(data, y)
    coeff3 = solver3.solve()
    y_pred3 = solver3.predict(data)
    mse3 = np.mean((y_pred3 - y)**2)
    
    # 求解器 4：核求解器（与求解器 2 数学上等价）
    solver4 = KernelSolver(kernel)
    solver4.load_data(data, y)
    coeff4 = solver4.solve()
    y_pred4 = solver4.predict(data)
    mse4 = np.mean((y_pred4 - y)**2)
    
    # 四种求解器都应给出合理的拟合结果（MSE 均小于 0.05）
    assert mse1 < 0.05
    assert mse2 < 0.05
    assert mse3 < 0.05
    assert mse4 < 0.05
    
    # 基方法的两套实现（FunctionalSolver + GramSolver）系数与预测应完全一致
    assert np.allclose(coeff1, coeff3, rtol=1e-10)
    assert np.allclose(y_pred1, y_pred3, rtol=1e-10)
    
    # 核方法的两套实现（FunctionalSolver + KernelSolver）系数与预测应完全一致
    assert np.allclose(coeff2, coeff4, rtol=1e-10)
    assert np.allclose(y_pred2, y_pred4, rtol=1e-10)

def test_solver_with_new_data():
    """Test prediction on new data."""
    # 训练数据：20 个带噪采样点
    x_train = np.linspace(0, 1, 20)
    y_train = np.sin(2 * np.pi * x_train) + 0.1 * np.random.randn(len(x_train))
    train_data = MultiDimData({0: x_train})
    
    # 测试数据：15 个新采样点（区间 [0.1, 0.9] 与训练集略有不同，用于检验泛化能力）
    x_test = np.linspace(0.1, 0.9, 15)
    y_test_true = np.sin(2 * np.pi * x_test)
    test_data = MultiDimData({0: x_test})
    
    # 用基方法（5 阶多项式）在训练集上训练求解器
    solver = FunctionalSolver()
    basis_set = BasisSet()
    for order in range(5):
        basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))
    solver.set_basis(basis_set)
    solver.set_inner_product(InnerProduct(is_continuous=False))
    solver.load_data(train_data, y_train)
    solver.solve()
    
    # 对从未见过的测试数据做预测
    y_test_pred = solver.predict(test_data)
    
    # 预测点的个数应与测试数据点数一致
    assert y_test_pred.shape == (15,)
    
    # 与无噪声的真值比较，计算测试集上的泛化误差
    test_mse = np.mean((y_test_pred - y_test_true)**2)
    assert test_mse < 0.2  # 泛化误差应在一个合理范围内

if __name__ == "__main__":
    # 直接运行本脚本时，顺序执行全部测试用例并打印通过信息
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
    
    print("\nAll solver tests passed!")
