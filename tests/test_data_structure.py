"""Tests for the data structures."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from geomfit.core.data_container import MultiDimData

def test_multi_dim_data_creation():
    """Test MultiDimData creation."""
    # 创建一维数据：维度 0 上有 3 个采样点
    data_1d = MultiDimData({0: np.array([1.0, 2.0, 3.0])})
    assert data_1d.n_dims == 1
    assert data_1d.n_points == 3
    assert 0 in data_1d.dims
    assert np.array_equal(data_1d.get_dim(0), np.array([1.0, 2.0, 3.0]))
    
    # 创建二维数据：维度 0 和维度 1 各有 3 个点（各维度长度必须一致）
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
    """Test retrieving individual data points."""
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([4.0, 5.0, 6.0]),
        2: np.array([7.0, 8.0, 9.0])
    })
    
    # 取出第 0 个点：应返回一个字典，键是维度号，值是该维度在第 0 个索引处的取值
    point0 = data.get_point(0)
    assert point0 == {0: 1.0, 1: 4.0, 2: 7.0}
    
    # 取出第 1 个点
    point1 = data.get_point(1)
    assert point1 == {0: 2.0, 1: 5.0, 2: 8.0}
    
    # 取出第 2 个点
    point2 = data.get_point(2)
    assert point2 == {0: 3.0, 1: 6.0, 2: 9.0}
    
    # 索引越界（索引 3 超出 0..2 的范围）时应抛出 IndexError
    try:
        data.get_point(3)
        assert False, "should raise IndexError"
    except IndexError:
        pass

def test_multi_dim_data_get_all_points():
    """Test retrieving all data points."""
    data = MultiDimData({
        0: np.array([1.0, 2.0]),
        1: np.array([3.0, 4.0])
    })
    
    # 该方法返回按点组织的列表：每个元素是一个 {维度号: 取值} 字典
    points = data.get_all_points()
    assert len(points) == 2
    assert points[0] == {0: 1.0, 1: 3.0}
    assert points[1] == {0: 2.0, 1: 4.0}

def test_multi_dim_data_validation():
    """Test data validation."""
    # 各维度长度不一致（维度 0 有 3 个点、维度 1 只有 2 个点）时应抛出 ValueError
    try:
        MultiDimData({
            0: np.array([1.0, 2.0, 3.0]),
            1: np.array([4.0, 5.0])  # 长度不一致（3 vs 2），用于触发校验错误
        })
        assert False, "should raise ValueError"
    except ValueError as e:
        assert "must have the same length" in str(e)
    
    # 空数据（不包含任何维度）时应抛出 ValueError
    try:
        MultiDimData({})
        assert False, "should raise ValueError"
    except ValueError as e:
        assert "must contain at least one dimension" in str(e)
    
    # 传入 Python 列表而不是 numpy 数组时应抛出 TypeError
    try:
        MultiDimData({0: [1.0, 2.0, 3.0]})  # 传的是列表而非 numpy 数组，用于触发类型校验
        assert False, "should raise TypeError"
    except TypeError as e:
        assert "must be numpy arrays" in str(e)

def test_multi_dim_data_values():
    """Test the values attribute."""
    data = MultiDimData({0: np.array([1.0, 2.0, 3.0])})
    
    # 未设置目标值时，values 属性应为 None
    assert data.values is None
    
    # 设置目标值：长度必须与数据点数（3）一致
    data.values = np.array([10.0, 20.0, 30.0])
    assert np.array_equal(data.values, np.array([10.0, 20.0, 30.0]))
    
    # 目标值长度（2）与数据点数（3）不一致时应抛出 ValueError
    try:
        data.values = np.array([10.0, 20.0])  # 长度与点数不一致，用于触发校验错误
        assert False, "should raise ValueError"
    except ValueError as e:
        assert "must have the same length" in str(e)

def test_multi_dim_data_operations():
    """Test arithmetic operations on data."""
    data1 = MultiDimData({0: np.array([1.0, 2.0, 3.0])})
    data2 = MultiDimData({0: np.array([4.0, 5.0, 6.0])})
    
    # 加法：逐点相加，即每个维度独立做 element-wise 加法
    data_sum = data1 + data2
    assert np.array_equal(data_sum.get_dim(0), np.array([5.0, 7.0, 9.0]))
    
    # 减法：data2 减去 data1
    data_diff = data2 - data1
    assert np.array_equal(data_diff.get_dim(0), np.array([3.0, 3.0, 3.0]))
    
    # 标量乘法：每个点的取值都乘以 2.0
    data_scaled = data1 * 2.0
    assert np.array_equal(data_scaled.get_dim(0), np.array([2.0, 4.0, 6.0]))
    
    # 标量除法：每个点的取值都除以 2.0
    data_divided = data2 / 2.0
    assert np.array_equal(data_divided.get_dim(0), np.array([2.0, 2.5, 3.0]))

def test_multi_dim_data_statistics():
    """Test summary statistics."""
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        1: np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    })
    
    # 均值：按维度分别计算（维度 1 恰好是维度 0 的两倍，均值也成比例）
    means = data.mean()
    assert means[0] == 3.0
    assert means[1] == 6.0
    
    # 标准差：按维度分别计算，并与 numpy 的 np.std 结果对比
    stds = data.std()
    assert np.allclose(stds[0], np.std([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.allclose(stds[1], np.std([2.0, 4.0, 6.0, 8.0, 10.0]))
    
    # 最小值与最大值：按维度分别统计
    mins = data.min()
    maxs = data.max()
    assert mins[0] == 1.0
    assert mins[1] == 2.0
    assert maxs[0] == 5.0
    assert maxs[1] == 10.0

def test_multi_dim_data_slicing():
    """Test slicing and masking."""
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        1: np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    })
    
    # 按索引列表切片：只保留索引 0、2、4 对应的点（即第 1、3、5 个点）
    sliced = data.slice(indices=[0, 2, 4])
    assert sliced.n_points == 3
    assert np.array_equal(sliced.get_dim(0), np.array([1.0, 3.0, 5.0]))
    assert np.array_equal(sliced.get_dim(1), np.array([6.0, 8.0, 10.0]))
    
    # 布尔掩码：True 的位置保留，False 的位置丢弃，效果应与上面的索引切片一致
    mask = np.array([True, False, True, False, True])
    masked = data.mask(mask)
    assert masked.n_points == 3
    assert np.array_equal(masked.get_dim(0), np.array([1.0, 3.0, 5.0]))
    assert np.array_equal(masked.get_dim(1), np.array([6.0, 8.0, 10.0]))

def test_multi_dim_data_concatenation():
    """Test concatenation."""
    data1 = MultiDimData({0: np.array([1.0, 2.0])})
    data2 = MultiDimData({0: np.array([3.0, 4.0])})
    
    # 沿点方向拼接（axis=0）：相当于把两个数据集的采样点首尾相连
    concatenated = MultiDimData.concatenate([data1, data2], axis=0)
    assert concatenated.n_points == 4
    assert np.array_equal(concatenated.get_dim(0), np.array([1.0, 2.0, 3.0, 4.0]))
    
    # 沿维度方向拼接（axis=1）：要求各数据集点数相同，拼接后维度数相加
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
    """Test saving and loading data."""
    import tempfile
    import pickle
    
    # 构造待保存的测试数据，并额外设置目标值 values
    original_data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([4.0, 5.0, 6.0])
    })
    original_data.values = np.array([10.0, 20.0, 30.0])
    
    # 用 pickle 序列化到临时文件（delete=False 表示关闭文件句柄后文件仍保留在磁盘上）
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        temp_file = f.name
        pickle.dump(original_data, f)
    
    try:
        # 从临时文件反序列化，恢复出数据对象
        with open(temp_file, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # 校验加载结果与保存前完全一致：维度数、点数、各维取值和目标值
        assert loaded_data.n_dims == 2
        assert loaded_data.n_points == 3
        assert np.array_equal(loaded_data.get_dim(0), np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(loaded_data.get_dim(1), np.array([4.0, 5.0, 6.0]))
        assert np.array_equal(loaded_data.values, np.array([10.0, 20.0, 30.0]))
    finally:
        # 清理临时文件，避免在磁盘上留下垃圾文件
        import os
        os.unlink(temp_file)

def test_multi_dim_data_from_array():
    """Test MultiDimData creation from an array."""
    # 一维数组：整体作为维度 dim=0 的数据
    array_1d = np.array([1.0, 2.0, 3.0])
    data_1d = MultiDimData.from_array(array_1d, dim=0)
    assert data_1d.n_dims == 1
    assert data_1d.n_points == 3
    assert np.array_equal(data_1d.get_dim(0), array_1d)
    
    # 二维数组：约定每一列对应一个维度（第 0 列 → dim 0，第 1 列 → dim 1）
    array_2d = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    data_2d = MultiDimData.from_array(array_2d, dims=[0, 1])
    assert data_2d.n_dims == 2
    assert data_2d.n_points == 3
    assert np.array_equal(data_2d.get_dim(0), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(data_2d.get_dim(1), np.array([4.0, 5.0, 6.0]))
    
    # 不显式指定维度时，自动按列顺序赋维度号 0, 1, ...
    data_auto = MultiDimData.from_array(array_2d)
    assert data_auto.n_dims == 2
    assert data_auto.dims == [0, 1]

def test_multi_dim_data_to_array():
    """Test conversion of MultiDimData to an array."""
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([4.0, 5.0, 6.0]),
        2: np.array([7.0, 8.0, 9.0])
    })
    
    # 把 MultiDimData 转回二维数组：形状为 (点数, 维度数)，每列是一个维度
    array = data.to_array()
    assert array.shape == (3, 3)  # 3 个点、3 个维度
    assert np.array_equal(array[:, 0], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(array[:, 1], np.array([4.0, 5.0, 6.0]))
    assert np.array_equal(array[:, 2], np.array([7.0, 8.0, 9.0]))
    
    # 自定义维度顺序 [2, 0, 1]：第 0 列应为原 dim 2 的数据，依此类推
    array_ordered = data.to_array(dims=[2, 0, 1])
    assert np.array_equal(array_ordered[:, 0], np.array([7.0, 8.0, 9.0]))
    assert np.array_equal(array_ordered[:, 1], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(array_ordered[:, 2], np.array([4.0, 5.0, 6.0]))

if __name__ == "__main__":
    # 直接运行本脚本时，顺序执行全部测试用例，每个用例通过后打印提示
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
    
    print("\nAll data structure tests passed!")
