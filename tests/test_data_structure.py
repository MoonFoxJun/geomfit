"""Tests for the data structures."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from functional_solver.core.data_container import MultiDimData

def test_multi_dim_data_creation():
    """Test MultiDimData creation."""
    # Create 1D data
    data_1d = MultiDimData({0: np.array([1.0, 2.0, 3.0])})
    assert data_1d.n_dims == 1
    assert data_1d.n_points == 3
    assert 0 in data_1d.dims
    assert np.array_equal(data_1d.get_dim(0), np.array([1.0, 2.0, 3.0]))
    
    # Create 2D data
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
    
    # Retrieve the first point
    point0 = data.get_point(0)
    assert point0 == {0: 1.0, 1: 4.0, 2: 7.0}
    
    # Retrieve the second point
    point1 = data.get_point(1)
    assert point1 == {0: 2.0, 1: 5.0, 2: 8.0}
    
    # Retrieve the third point
    point2 = data.get_point(2)
    assert point2 == {0: 3.0, 1: 6.0, 2: 9.0}
    
    # An out-of-range index should raise IndexError
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
    
    points = data.get_all_points()
    assert len(points) == 2
    assert points[0] == {0: 1.0, 1: 3.0}
    assert points[1] == {0: 2.0, 1: 4.0}

def test_multi_dim_data_validation():
    """Test data validation."""
    # Dimensions of mismatched lengths should raise ValueError
    try:
        MultiDimData({
            0: np.array([1.0, 2.0, 3.0]),
            1: np.array([4.0, 5.0])  # mismatched lengths
        })
        assert False, "should raise ValueError"
    except ValueError as e:
        assert "must have the same length" in str(e)
    
    # Empty data should raise ValueError
    try:
        MultiDimData({})
        assert False, "should raise ValueError"
    except ValueError as e:
        assert "must contain at least one dimension" in str(e)
    
    # Non-array data should raise TypeError
    try:
        MultiDimData({0: [1.0, 2.0, 3.0]})  # a list instead of a numpy array
        assert False, "should raise TypeError"
    except TypeError as e:
        assert "must be numpy arrays" in str(e)

def test_multi_dim_data_values():
    """Test the values attribute."""
    data = MultiDimData({0: np.array([1.0, 2.0, 3.0])})
    
    # values should be None initially
    assert data.values is None
    
    # Set values
    data.values = np.array([10.0, 20.0, 30.0])
    assert np.array_equal(data.values, np.array([10.0, 20.0, 30.0]))
    
    # A mismatched length should raise ValueError
    try:
        data.values = np.array([10.0, 20.0])  # mismatched length
        assert False, "should raise ValueError"
    except ValueError as e:
        assert "must have the same length" in str(e)

def test_multi_dim_data_operations():
    """Test arithmetic operations on data."""
    data1 = MultiDimData({0: np.array([1.0, 2.0, 3.0])})
    data2 = MultiDimData({0: np.array([4.0, 5.0, 6.0])})
    
    # Addition
    data_sum = data1 + data2
    assert np.array_equal(data_sum.get_dim(0), np.array([5.0, 7.0, 9.0]))
    
    # Subtraction
    data_diff = data2 - data1
    assert np.array_equal(data_diff.get_dim(0), np.array([3.0, 3.0, 3.0]))
    
    # Scalar multiplication
    data_scaled = data1 * 2.0
    assert np.array_equal(data_scaled.get_dim(0), np.array([2.0, 4.0, 6.0]))
    
    # Scalar division
    data_divided = data2 / 2.0
    assert np.array_equal(data_divided.get_dim(0), np.array([2.0, 2.5, 3.0]))

def test_multi_dim_data_statistics():
    """Test summary statistics."""
    data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        1: np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    })
    
    # Mean
    means = data.mean()
    assert means[0] == 3.0
    assert means[1] == 6.0
    
    # Standard deviation
    stds = data.std()
    assert np.allclose(stds[0], np.std([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.allclose(stds[1], np.std([2.0, 4.0, 6.0, 8.0, 10.0]))
    
    # Minimum and maximum
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
    
    # Slice by index list
    sliced = data.slice(indices=[0, 2, 4])
    assert sliced.n_points == 3
    assert np.array_equal(sliced.get_dim(0), np.array([1.0, 3.0, 5.0]))
    assert np.array_equal(sliced.get_dim(1), np.array([6.0, 8.0, 10.0]))
    
    # Boolean mask
    mask = np.array([True, False, True, False, True])
    masked = data.mask(mask)
    assert masked.n_points == 3
    assert np.array_equal(masked.get_dim(0), np.array([1.0, 3.0, 5.0]))
    assert np.array_equal(masked.get_dim(1), np.array([6.0, 8.0, 10.0]))

def test_multi_dim_data_concatenation():
    """Test concatenation."""
    data1 = MultiDimData({0: np.array([1.0, 2.0])})
    data2 = MultiDimData({0: np.array([3.0, 4.0])})
    
    # Concatenate along points (axis=0)
    concatenated = MultiDimData.concatenate([data1, data2], axis=0)
    assert concatenated.n_points == 4
    assert np.array_equal(concatenated.get_dim(0), np.array([1.0, 2.0, 3.0, 4.0]))
    
    # Concatenate along dimensions (axis=1; requires equal point counts)
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
    
    # Create test data
    original_data = MultiDimData({
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([4.0, 5.0, 6.0])
    })
    original_data.values = np.array([10.0, 20.0, 30.0])
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        temp_file = f.name
        pickle.dump(original_data, f)
    
    try:
        # Load from file
        with open(temp_file, 'rb') as f:
            loaded_data = pickle.load(f)
        
        # Verify the loaded data
        assert loaded_data.n_dims == 2
        assert loaded_data.n_points == 3
        assert np.array_equal(loaded_data.get_dim(0), np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(loaded_data.get_dim(1), np.array([4.0, 5.0, 6.0]))
        assert np.array_equal(loaded_data.values, np.array([10.0, 20.0, 30.0]))
    finally:
        # Clean up the temporary file
        import os
        os.unlink(temp_file)

def test_multi_dim_data_from_array():
    """Test MultiDimData creation from an array."""
    # 1D array
    array_1d = np.array([1.0, 2.0, 3.0])
    data_1d = MultiDimData.from_array(array_1d, dim=0)
    assert data_1d.n_dims == 1
    assert data_1d.n_points == 3
    assert np.array_equal(data_1d.get_dim(0), array_1d)
    
    # 2D array (each column is a dimension)
    array_2d = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    data_2d = MultiDimData.from_array(array_2d, dims=[0, 1])
    assert data_2d.n_dims == 2
    assert data_2d.n_points == 3
    assert np.array_equal(data_2d.get_dim(0), np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(data_2d.get_dim(1), np.array([4.0, 5.0, 6.0]))
    
    # Dimensions are assigned automatically when omitted
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
    
    # Convert to an array
    array = data.to_array()
    assert array.shape == (3, 3)  # 3 points, 3 dimensions
    assert np.array_equal(array[:, 0], np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(array[:, 1], np.array([4.0, 5.0, 6.0]))
    assert np.array_equal(array[:, 2], np.array([7.0, 8.0, 9.0]))
    
    # Custom dimension ordering
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
    
    print("\nAll data structure tests passed!")
