import pandas as pd
import pytest

def df_append(df1, df2, after=None, remove=False):
    """
    Append columns from df2 to df1.

    Parameters:
    df1: pandas DataFrame
    df2: pandas DataFrame or dict-like
    after: int or str, optional
        Position after which to append the columns from df2.
        If int, insert after the 0-based position.
        If str, insert after the column with this name.
        If None, append at the end.
    remove: bool, default False
        If True, remove columns from df1 that have the same names as in df2 before appending.
    """
    df1 = df1.copy()

    if isinstance(df2, dict):
        cols_to_add = df2
    else:
        cols_to_add = df2

    if isinstance(cols_to_add, pd.DataFrame):
        cols_to_insert = cols_to_add
    else:
        cols_to_insert = pd.DataFrame(cols_to_add)

    df2_cols = cols_to_insert.columns

    if remove:
        # Remove columns in df1 that are in df2_cols
        df1 = df1.drop(columns=[col for col in df2_cols if col in df1.columns], errors='ignore')
    else:
        # Overwrite existing columns in df1 with those from df2
        pass  # Existing columns will be overwritten

    if after is None:
        # Append at the end
        for col in df2_cols:
            df1[col] = cols_to_insert[col].values
    else:
        if isinstance(after, int):
            if not isinstance(after, int) or after < 0:
                raise ValueError("`after` must be a non-negative integer")
            position = after
        elif isinstance(after, str):
            if after not in df1.columns:
                raise ValueError(f"Column '{after}' not found in df1")
            position = df1.columns.get_loc(after)
        else:
            raise ValueError("`after` must be integer or string")

        position += 1  # Since we insert after the given position

        # Overwrite existing columns or add new ones
        for col in df2_cols:
            df1[col] = cols_to_insert[col].values

        # Create new column order
        cols_before = df1.columns[:position]
        cols_after = df1.columns[position:]
        new_cols = list(cols_before) + list(df2_cols) + [col for col in cols_after if col not in df2_cols]

        df1 = df1[new_cols]
    return df1

def test_columns_in_y_replace_those_in_x():
    df1 = pd.DataFrame({'x': [1]})
    df2 = pd.DataFrame({'x': [2]})

    assert df_append(df1, df2).equals(df2)

def test_replaced_columns_retain_correct_ordering_1444():
    df1 = pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})
    df2 = pd.DataFrame({'x': [4]})

    expected1 = pd.DataFrame({'x': [4], 'y': [2], 'z': [3]})
    expected2 = pd.DataFrame({'x': [4], 'y': [2], 'z': [3]})
    expected3 = pd.DataFrame({'y': [2], 'x': [4], 'z': [3]})

    result1 = df_append(df1, df2, after=0)
    result2 = df_append(df1, df2, after=1)
    result3 = df_append(df1, df2, after=2)

    pd.testing.assert_frame_equal(result1, expected1)
    pd.testing.assert_frame_equal(result2, expected2)
    pd.testing.assert_frame_equal(result3, expected3)

def test_after_must_be_integer_or_character():
    df1 = pd.DataFrame({'x': [1]})
    df2 = pd.DataFrame({'x': [2]})

    with pytest.raises(ValueError):
        df_append(df1, df2, after=1.5)

def test_always_returns_a_bare_data_frame():
    df1 = pd.DataFrame({'x': [1]})
    df2 = pd.DataFrame({'y': [2]})

    result = df_append(df1, df2)
    expected = pd.DataFrame({'x': [1], 'y': [2]})

    pd.testing.assert_frame_equal(result, expected)
    assert isinstance(result, pd.DataFrame)

def test_retains_row_names_of_dataframe_x_1454():
    # These can't be restored by `reconstruct_tibble()`, so it is reasonable to
    # retain them. `dplyr:::dplyr_col_modify()` works similarly.
    df = pd.DataFrame({'x': [1, 2]}, index=['a', 'b'])
    cols = {'y': [3, 4], 'z': [5,6]}

    result1 = df_append(df, cols)
    result2 = df_append(df, cols, after=0)
    result3 = df_append(df, cols, remove=True)

    assert list(result1.index) == ['a', 'b']
    assert list(result2.index) == ['a', 'b']
    assert list(result3.index) == ['a', 'b']

def test_can_append_at_any_integer_position():
    df1 = pd.DataFrame({'x': [1], 'y': [2]})
    df2 = pd.DataFrame({'a': [1]})

    result0 = df_append(df1, df2, 0)
    result1 = df_append(df1, df2, 1)
    result2 = df_append(df1, df2, 2)

    assert list(result0.columns) == ['a', 'x', 'y']
    assert list(result1.columns) == ['x', 'a', 'y']
    assert list(result2.columns) == ['x', 'y', 'a']

def test_can_append_at_any_character_position():
    df1 = pd.DataFrame({'x': [1], 'y': [2]})
    df2 = pd.DataFrame({'a': [1]})

    result_x = df_append(df1, df2, 'x')
    result_y = df_append(df1, df2, 'y')

    assert list(result_x.columns) == ['x', 'a', 'y']
    assert list(result_y.columns) == ['x', 'y', 'a']

def test_can_replace_at_any_character_position():
    df1 = pd.DataFrame({'x': [1], 'y': [2], 'z': [3]})
    df2 = pd.DataFrame({'a': [1]})

    result_x = df_append(df1, df2, 'x', remove=True)
    result_y = df_append(df1, df2, 'y', remove=True)
    result_z = df_append(df1, df2, 'z', remove=True)

    assert list(result_x.columns) == ['a', 'y', 'z']
    assert list(result_y.columns) == ['x', 'a', 'z']
    assert list(result_z.columns) == ['x', 'y', 'a']