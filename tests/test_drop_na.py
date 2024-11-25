import pandas as pd
import pandas.testing as pd_testing
import pytest
from tidyr_py_openai import drop_na

def starts_with(prefix, df):
    return [col for col in df.columns if col.startswith(prefix)]

def test_empty_call_drops_every_row():
    # Test that an empty call drops every row with any missing values
    df = pd.DataFrame({
    'x': pd.Series([1, 2, None], dtype='Int64'),  # Nullable integer type
    'y': pd.Series(['a', None, 'b'], dtype='string')  # Nullable string type
})

    #df = pd.DataFrame({'x': [1, 2, None],
    #                   'y': ['a', None, 'b']})
    exp = pd.DataFrame({'x': pd.Series([1],dtype='Int64'),
                        'y': pd.Series(['a'],dtype='string')
                        })
    res = drop_na(df)
    pd_testing.assert_frame_equal(res.reset_index(drop=True), exp.reset_index(drop=True))

def test_tidyselection_selects_no_columns_doesnt_drop_any_rows():
    # Test that selecting no columns doesn't drop any rows
    df = pd.DataFrame({
        'x': pd.Series([1, 2, None],dtype='Int64'),
        'y': pd.Series(['a', None, 'b'],dtype='string')
    })
    res = drop_na(df, starts_with('foo', df),allow_empty_params=True)
    pd_testing.assert_frame_equal(res.reset_index(drop=True), df.reset_index(drop=True))

def test_specifying_variables_considers_only_that_variable():
    # Test that specifying variables considers only those variables
    df = pd.DataFrame({
        'x': pd.Series([1, 2, None],dtype='Int64'),
        'y': pd.Series(['a', None, 'b'],dtype='string')
    })
    exp = pd.DataFrame({
        'x': pd.Series([1, 2],dtype='Int64'),
        'y': pd.Series(['a', None], dtype='string')
                       })
    res = drop_na(df, 'x')
    pd_testing.assert_frame_equal(res.reset_index(drop=True), exp.reset_index(drop=True))

    # Define a function to select a range of columns
    def col_range(start_col, end_col, df):
        cols = df.columns.tolist()
        start_idx = cols.index(start_col)
        end_idx = cols.index(end_col)
        return cols[start_idx:end_idx+1]

    exp = pd.DataFrame({
        'x': pd.Series([1],dtype='Int64'),
        'y': pd.Series(['a'],dtype='string')
    })
    res = drop_na(df, *col_range('x', 'y', df))
    pd_testing.assert_frame_equal(res.reset_index(drop=True), exp.reset_index(drop=True))

def test_errors_are_raised():
    # Test that errors are raised for invalid inputs
    df = pd.DataFrame({
        'x': pd.Series([1, 2, None],dtype='Int64'),
        'y': pd.Series(['a', None, 'b'],dtype='string')
                       })
    with pytest.raises(ValueError):
        drop_na(df, [])
    with pytest.raises(KeyError):
        drop_na(df, 'z')

def test_single_variable_data_frame_doesnt_lose_dimension():
    # Test that single variable DataFrame doesn't lose dimension
    df = pd.DataFrame({
        'x': pd.Series([1, 2, None],dtype='Int64')
    })
    res = drop_na(df, 'x')
    exp = pd.DataFrame({'x': pd.Series([1, 2],dtype='Int64')
                                       })
    pd_testing.assert_frame_equal(res.reset_index(drop=True), exp.reset_index(drop=True))

def test_works_with_list_cols():
    # Test that drop_na works with list-columns
    df = pd.DataFrame({
        'x': pd.Series([[1], None, [3]]),
        'y': pd.Series([1, 2, None],dtype='Int64')
                       })
    res = drop_na(df)
    exp = pd.DataFrame({
        'x': pd.Series([[1]]),
        'y': pd.Series([1],dtype='Int64')
                       })
    pd_testing.assert_frame_equal(res.reset_index(drop=True), exp.reset_index(drop=True))

def test_doesnt_drop_empty_atomic_elements_of_list_cols():
    # Test that empty atomic elements of list-columns are not dropped
    df = pd.DataFrame({
        'x': pd.Series([[1], None, []])
    })
    res = drop_na(df)
    exp = df.iloc[[0, 2]].reset_index(drop=True)
    pd_testing.assert_frame_equal(res.reset_index(drop=True), exp)

def test_preserves_attributes():
    #new code
    # Test that attributes are preserved
    s = pd.Series([1, None],dtype='Int64')
    s.attrs['attr'] = '!'
    df = pd.DataFrame({'x': s})
    res = drop_na(df)
    exp_s = pd.Series([1],dtype='Int64')
    exp_s.attrs['attr'] = '!'
    exp = pd.DataFrame({'x': exp_s})
    assert res['x'].attrs == exp['x'].attrs
    pd_testing.assert_frame_equal(res.reset_index(drop=True), exp.reset_index(drop=True))
