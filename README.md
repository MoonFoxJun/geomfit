# Functional Solver - 泛函解算器

一个用于求解泛函优化问题的Python库，支持基函数展开和核方法。

## 功能特性

- **多维数据支持**：处理任意维度的数据点
- **多维张量积基**：f(x,y,z) = Σᵢ Σⱼ Σₖ cᵢⱼₖ Xᵢ(x)Yⱼ(y)Zₖ(z)，完整表达维度间的交互/乘积结构（不是简单的各维直和）
- **灵活的基函数**：多项式、傅里叶、小波、径向基函数等，可组合成张量积
- **核方法**：RBF核、多项式核、Matern核等（核天然是多维的，不受维数直和限制）
- **可配置内积**：支持权重函数和正则化，多维连续内积带体积元
- **工程化结构**：模块化设计，易于扩展和维护

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/functional_solver.git
cd functional_solver

# （推荐）创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows；Linux/macOS: source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

## 快速开始

```python
import numpy as np
from functional_solver import MultiDimData, BasisSet, BasisFactory, InnerProduct, FunctionalSolver

# 1. 准备数据
x = np.linspace(0, 1, 50)
y_true = np.sin(2 * np.pi * x)
y = y_true + 0.1 * np.random.randn(50)

data = MultiDimData({0: x})

# 2. 构建基函数
basis_set = BasisSet()
for order in range(5):
    basis_set.add_basis(BasisFactory.polynomial(dim=0, order=order))

# 3. 设置内积
inner = InnerProduct(weight_func=lambda x: np.exp(-x))

# 4. 求解
solver = FunctionalSolver()
solver.set_basis(basis_set)
solver.set_inner_product(inner)
solver.load_data(data, y)
coeff = solver.solve(regularization={"method": "tikhonov", "alpha": 1e-6})

# 5. 预测
y_pred = solver.predict(data)
```

### 多维张量积基（推荐）

多维时请用 `BasisFactory.tensor_product` 构造乘积基，而不是把各维基函数
简单堆进同一个集合（那是直和，无法表示交互项如 x·y）：

```python
import numpy as np
from functional_solver import MultiDimData, BasisSet, BasisFactory, InnerProduct, FunctionalSolver

# 1. 二维数据（真函数含交互项 x*y）
x = np.random.uniform(0, 1, 50)
y = np.random.uniform(0, 1, 50)
z = 1 + 2*x - y + 3*x*y + 0.1 * np.random.randn(50)
data = MultiDimData({0: x, 1: y})

# 2. 张量积基：3×3 = 9 个 Xᵢ(x)Yⱼ(y)
x_bases = [BasisFactory.polynomial(dim=0, order=o) for o in range(3)]
y_bases = [BasisFactory.polynomial(dim=1, order=o) for o in range(3)]
basis_set = BasisSet()
for b in BasisFactory.tensor_product({0: x_bases, 1: y_bases}):
    basis_set.add_basis(b)

# 3. 求解（与一维完全相同的流程）
solver = FunctionalSolver()
solver.set_basis(basis_set)
solver.set_inner_product(InnerProduct(is_continuous=False))
solver.load_data(data, z)
coeff = solver.solve()
z_pred = solver.predict(data)
```

### 直和（加法模型）保留模块

高维数据（张量积基数量随维数指数增长，维数灾难）或各维度贡献确实独立时，
可使用保留的直和实现 [`functional_solver/basis/additive.py`](functional_solver/basis/additive.py)
（`AdditiveBasis.build`）：它把各维基函数拼接成加法模型
f(x) = Σ_d Σ_j c_{d,j} φ_{d,j}(x_d)，并自动去重跨维度重复的常数基。
其数学局限（无法表示交互项、完备性差等）详见该模块文档。

### 强相关维度的处理：PCA 解耦 + 马氏距离核

张量积基假设坐标正交；当数据集中在窄带（如 y≈x，强相关）时，各维基列近共线，
Gram 矩阵病态、系数巨大。两个标准解法（已内置）：

```python
from functional_solver.utils.preprocess import pca_rotate, pca_transform
from functional_solver.kernel.mahalanobis import MahalanobisKernel

# 1) PCA 旋转（可选丢弃近零方差方向）后再做张量积
rotated, info = pca_rotate(data, n_components=1)   # 主轴方向方差占比≈1
# ... 在 rotated 上构造张量积基并求解 ...
new_rotated = pca_transform(new_data, info)        # 新数据用同一变换

# 2) 或直接用马氏距离核（把相关性吸收进距离度量；核方法记得配岭正则）
kernel = MahalanobisKernel(sigma=1.0)              # metric 自动从数据估计
solver.set_kernel(kernel)
solver.solve(regularization={"alpha": 1e-6})
```

量化对比见 [`examples/demo_correlated_dims.py`](examples/demo_correlated_dims.py)
（条件数：原始坐标张量积 ≈1e9 vs PCA 解耦后 ≈8.6；系数范数 140 vs 1）。

## 项目结构

```
functional_solver/
├── functional_solver/       # 包本体（import functional_solver.*）
│   ├── core/                # 核心数据结构（MultiDimData、BasisSet）
│   ├── basis/               # 基函数（多项式/傅里叶/小波/RBF；含 tensor_product 张量积、additive 直和）
│   ├── kernel/              # 核函数（RBF/Matern/多项式/复合核）
│   ├── inner_product/       # 内积与正则化
│   ├── solver/              # 解算器（FunctionalSolver 等 5 个）
│   └── utils/               # 数值/可视化/日志工具
├── examples/                # 示例脚本（相当于主程序）
├── tests/                   # 测试代码（pytest）
├── docs/                    # 文档（code_map.md、architecture.md）
└── output/                  # 示例运行生成的图片（已 gitignore）
```

## 文档

- [docs/architecture.md](docs/architecture.md) —— 架构速览（5 分钟看懂文件关系）
- [docs/code_map.md](docs/code_map.md) —— 代码地图（每个模块对应什么数学概念）
- [docs/kernel_convolution_attention.md](docs/kernel_convolution_attention.md) —— 卷积/核/注意力三者的数学关系（10 分钟）
- 配套演示：[examples/demo_attention_as_kernel.py](examples/demo_attention_as_kernel.py)（QKᵀ ≡ 核矩阵，精确恒等式）

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_basis.py
```

## 许可证

MIT License
