"""测试数据结构"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from functional_solver.core.data_container import MultiDimData

def test_multi_dim_data_creation():
    """测试MultiDimData创建"""
    # 创建1D数据
    data_1d = MultiDimData({0: np.array([1.0, 2.0, 3.0])})
    assert data_1d.n_dims == 1
    assert data_1d.n_points == 3
    assert 0 in data_1d.dims
    assert np.array_equal(data_1d.get_dim(0), np.array([1.0, 2.0, 3.0]))
    
    # 创建2D数据
    data_2d = MultiDimData({
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([4.0, 5.0, 6.0])
    })
    assert data_2d.n_dims == 2
    assert data_2d.n_points == 3
    assert 0 in data_2d.dims
    assert 1 in data_2d.dims
    assert np.array_equal(data_2d.get_dim(0), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(data_2d.get_dim(1), np.array([4.0, 5.0, 6.0]))

def test_multi_dim_data_get_point():
    """测试获取数据点"""
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([4.0, 5.0, 6.0]),
        2: np.array([7.0, 8.0, 9.0])
    })
    
    # 测试获取第一个点
    point0 = data.get_point(0)
    assert point0 == {0: 1.0, 1: 4.0, 2: 7.0}
    
    # 测试获取第二个点
    point1 = data.get_point(1)
    assert point1 == {0: 2.0, 1: 5.0, 2: 8.0}
    
    # 测试获取第三个点
    point2 = data.get_point(2)
    assert point2 == {0: 3.0, 1: 6.0, 2: 9.0}
    
    # 测试索引越界
    try:
        data.get_point(3)
        assert False, "应该抛出IndexError"
    except IndexError:
        pass

def test_multi_dim_data_get_all_points():
    """测试获取所有数据点"""
    data = MultiDimData({
        0: np.array([1.0, 2.0]),
        1: np.array([3.0, 4.0])
    })
    
    points = data.get_all_points()
    assert len(points) == 2
    assert points[0] == {0: 1.0, 1: 3.0}
    assert points[1] == {0: 2.0, 1: 4.0}

def test_multi_dim_data_validation():
    """测试数据验证"""
    # 测试不同长度的维度数据
    try:
        MultiDimData({
            0: np.array([1.0, 2.0, 3.0]),
            1: np.array([4.0, 5.0])  # 长度不一致
        })
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "must have the same length" in str(e)
    
    # 测试空数据
    try:
        MultiDimData({})
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "must contain at least one dimension" in str(e)
    
    # 测试非数组数据
    try:
        MultiDimData({0: [1.0, 2.0, 3.0]})  # 列表而不是numpy数组
        assert False, "应该抛出TypeError"
    except TypeError as e:
        assert "must be numpy arrays" in str(e)

def test_multi_dim_data_values():
    """测试数据值"""
    data = MultiDimData({0: np.array([1.0, 2.0, 3.0])})
    
    # 初始时values应该为None
    assert data.values is None
    
    # 设置values
    data.values = np.array([10.0, 20.0, 30.0])
    assert np.array_equal(data.values, np.array([10.0, 20.0, 30.0]))
    
    # 测试values长度验证
    try:
        data.values = np.array([10.0, 20.0])  # 长度不一致
        assert False, "应该抛出ValueError"
    except ValueError as e:
        assert "must have the same length" in str(e)

def test_multi_dim_data_operations():
    """测试数据操作"""
    data1 = MultiDimData({0: np.array([1.0, 2.0, 3.0])})
    data2 = MultiDimData({0: np.array([4.0, 5.0, 6.0])})
    
    # 测试加法
    data_sum = data1 + data2
    assert np.array_equal(data_sum.get_dim(0), np.array([5.0, 7.0, 9.0]))
    
    # 测试减法
    data_diff = data2 - data1
    assert np.array_equal(data_diff.get_dim(0), np.array([3.0, 3.0, 3.0]))
    
    # 测试标量乘法
    data_scaled = data1 * 2.0
    assert np.array_equal(data_scaled.get_dim(0), np.array([2.0, 4.0, 6.0]))
    
    # 测试标量除法
    data_divided = data2 / 2.0
    assert np.array_equal(data_divided.get_dim(0), np.array([2.0, 2.5, 3.0]))

def test_multi_dim_data_statistics():
    """测试数据统计"""
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        1: np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    })
    
    # 测试均值
    means = data.mean()
    assert means[0] == 3.0
    assert means[1] == 6.0
    
    # 测试标准差
    stds = data.std()
    assert np.allclose(stds[0], np.std([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.allclose(stds[1], np.std([2.0, 4.0, 6.0, 8.0, 10.0]))
    
    # 测试最小值和最大值
    mins = data.min()
    maxs = data.max()
    assert mins[0] == 1.0
    assert mins[1] == 2.0
    assert maxs[0] == 5.0
    assert maxs[1] == 10.0

def test_multi_dim_data_slicing():
    """测试数据切片"""
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        1: np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    })
    
    # 测试切片
    sliced = data.slice(indices=[0, 2, 4])
    assert sliced.n_points == 3
    assert np.array_equal(sliced.get_dim(0), np.array([1.0, 3.0, 5.0]))
    assert np.array_equal(sliced.get_dim(1), np.array([6.0, 8.0, 10.0]))
    
    # 测试布尔掩码
    mask = np.array([True, False, True, False, True])
    masked = data.mask(mask)
    assert masked.n_points == 3
    assert np.array_equal(masked.get_dim(0), np.array([1.0, 3.0, 5.0]))
    assert np.array_equal(masked.get_dim(1), np.array([6.0, 8.0, 10.0]))

def test_multi_dim_data_concatenation():
    """测试数据拼接"""
    data1 = MultiDimData({0: np.array([1.0, 2.0])})
    data2 = MultiDimData({0: np.array([3.0, 4.0])})
    
    # 测试垂直拼接
    concatenated = MultiDimData.concatenate([data1, data2], axis=0)
    assert concatenated.n_points == 4
    assert np.array_equal(concatenated.get_dim(0), np.array([1.0, 2.0, 3.0, 4.0]))
    
    # 测试水平拼接（需要相同数据点数量）
    data3 = MultiDimData({
        0: np.array([1.0, 2.0]),
        1: np.array([3.0, 4.0])
    })
    data4 = MultiDimData({
        2: np.array([5.0, 6.0]),
        3: np.array([7.0, 8.0])
    })
    
    concatenated_h = MultiDimData.concatenate([data3, data4], axis=1)
    assert concatenated_h.n_dims == 4
    assert concatenated_h.n_points == 2
    assert np.array_equal(concatenated_h.get_dim(0), np.array([1.0, 2.0]))
    assert np.array_equal(concatenated_h.get_dim(3), np.array([7.0, 8.0]))

def test_multi_dim_data_save_load():
    """测试数据保存和加载"""
    import tempfile
    import pickle
    
    # 创建测试数据
    original_data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([4.0, 5.0, 6.0])
    })
    original_data.values = np.array([10.0, 20.0, 30.0])
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        temp_file = f.name
        pickle.dump(original_data, f)
    
    try:
        # 从文件加载
        with open(temp_file, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # 验证加载的数据
        assert loaded_data.n_dims == 2
        assert loaded_data.n_points == 3
        assert np.array_equal(loaded_data.get_dim(0), np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(loaded_data.get_dim(1), np.array([4.0, 5.0, 6.0]))
        assert np.array_equal(loaded_data.values, np.array([10.0, 20.0, 30.0]))
    finally:
        # 清理临时文件
        import os
        os.unlink(temp_file)

def test_multi_dim_data_from_array():
    """测试从数组创建MultiDimData"""
    # 1D数组
    array_1d = np.array([1.0, 2.0, 3.0])
    data_1d = MultiDimData.from_array(array_1d, dim=0)
    assert data_1d.n_dims == 1
    assert data_1d.n_points == 3
    assert np.array_equal(data_1d.get_dim(0), array_1d)
    
    # 2D数组（每列是一个维度）
    array_2d = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    data_2d = MultiDimData.from_array(array_2d, dims=[0, 1])
    assert data_2d.n_dims == 2
    assert data_2d.n_points == 3
    assert np.array_equal(data_2d.get_dim(0), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(data_2d.get_dim(1), np.array([4.0, 5.0, 6.0]))
    
    # 测试自动分配维度
    data_auto = MultiDimData.from_array(array_2d)
    assert data_auto.n_dims == 2
    assert data_auto.dims == [0, 1]

def test_multi_dim_data_to_array():
    """测试MultiDimData转换为数组"""
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([4.0, 5.0, 6.0]),
        2: np.array([7.0, 8.0, 9.0])
    })
    
    # 转换为数组
    array = data.to_array()
    assert array.shape == (3, 3)  # 3个点，3个维度
    assert np.array_equal(array[:, 0], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(array[:, 1], np.array([4.0, 5.0, 6.0]))
    assert np.array_equal(array[:, 2], np.array([7.0, 8.0, 9.0]))
    
    # 测试指定维度顺序
    array_ordered = data.to_array(dims=[2, 0, 1])
    assert np.array_equal(array_ordered[:, 0], np.array([7.0, 8.0, 9.0]))
    assert np.array_equal(array_ordered[:, 1], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(array_ordered[:, 2], np.array([4.0, 5.0, 6.0]))

if __name__ == "__main__":
    test_multi_dim_data_creation()
    print("✓ test_multi_dim_data_creation passed")
    
    test_multi_dim_data_get_point()
    print("✓ test_multi_dim_data_get_point passed")
    
    test_multi_dim_data_get_all_points()
    print("✓ test_multi_dim_data_get_all_points passed")
    
    test_multi_dim_data_validation()
    print("✓ test_multi_dim_data_validation passed")
    
    test_multi_dim_data_values()
    print("✓ test_multi_dim_data_values passed")
    
    test_multi_dim_data_operations()
    print("✓ test_multi_dim_data_operations passed")
    
    test_multi_dim_data_statistics()
    print("✓ test_multi_dim_data_statistics passed")
    
    test_multi_dim_data_slicing()
    print("✓ test_multi_dim_data_slicing passed")
    
    test_multi_dim_data_concatenation()
    print("✓ test_multi_dim_data_concatenation passed")
    
    test_multi_dim_data_save_load()
    print("✓ test_multi_dim_data_save_load passed")
    
    test_multi_dim_data_from_array()
    print("✓ test_multi_dim_data_from_array passed")
    
    test_multi_dim_data_to_array()
    print("✓ test_multi_dim_data_to_array passed")
    
    print("\n所有数据结构测试通过！")
