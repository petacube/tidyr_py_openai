import pandas as pd
import pytest

def pack(df, names_sep=None, **kwargs):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if not kwargs:
        return df.copy()
    if names_sep is not None and not isinstance(names_sep, str):
        raise ValueError("`names_sep` must be a string or None.")

    new_df = df.copy()
    for new_col, cols in kwargs.items():
        if not isinstance(cols, list):
            raise ValueError("Columns to pack must be a list of column names.")
        for col in cols:
            if not isinstance(col, str):
                raise ValueError("Columns to pack must be specified as strings.")
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        sub_df = df[cols].copy()
        if names_sep is not None:
            prefix = new_col + names_sep
            sub_df.columns = [col[len(prefix):] if col.startswith(prefix) else col for col in sub_df.columns]
        new_df[new_col] = sub_df.to_dict(orient='records')
        new_df = new_df.drop(columns=cols)
    return new_df

def unpack(df, columns=None, names_sep=None):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if columns is None:
        raise ValueError("At least one column must be specified for unpacking.")
    if names_sep is not None and not isinstance(names_sep, str):
        raise ValueError("`names_sep` must be a string or None.")
    if not isinstance(columns, list):
        columns = [columns]
    new_df = df.copy()
    for col in columns:
        if col not in new_df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if not new_df[col].apply(lambda x: isinstance(x, dict)).all():
            continue
        unpacked = pd.DataFrame(new_df[col].tolist())
        if names_sep:
            unpacked.columns = [f"{col}{names_sep}{c}" for c in unpacked.columns]
        for c in unpacked.columns:
            if c in new_df.columns:
                raise ValueError(f"Column '{c}' already exists.")
        new_df = new_df.drop(columns=[col])
        new_df = pd.concat([new_df, unpacked], axis=1)
    return new_df

# pack --------------------------------------------------------------------

def test_can_pack_multiple_columns():
    df = pd.DataFrame({'a1': [1], 'a2': [2], 'b1': [1], 'b2': [2]})
    out = pack(df, a=['a1', 'a2'], b=['b1', 'b2'])
    assert list(out.columns) == ['a', 'b']
    pd.testing.assert_frame_equal(pd.DataFrame(out['a'].tolist()), df[['a1', 'a2']])
    pd.testing.assert_frame_equal(pd.DataFrame(out['b'].tolist()), df[['b1', 'b2']])

def test_packing_no_columns_returns_input():
    df = pd.DataFrame({'a1': [1], 'a2': [2], 'b1': [1], 'b2': [2]})
    out = pack(df)
    pd.testing.assert_frame_equal(out, df)

def test_can_strip_outer_names_from_inner_names():
    df = pd.DataFrame({'ax': [1], 'ay': [2]})
    out = pack(df, a=['ax', 'ay'], names_sep="")
    a_df = pd.DataFrame(out['a'].tolist())
    assert list(a_df.columns) == ['x', 'y']

def test_pack_disallows_renaming():
    df = pd.DataFrame({'x': [1], 'y': [2]})
    with pytest.raises(ValueError):
        pack(df, data={'a': 'x'})
    with pytest.raises(ValueError):
        pack(df, data1='x', data2={'a': 'y'})

def test_pack_validates_its_inputs():
    df = pd.DataFrame({'a1': [1], 'a2': [2], 'b1': [1], 'b2': [2]})
    with pytest.raises(TypeError):
        pack(1)
    with pytest.raises(TypeError):
        pack(df, ['a1', 'a2'], ['b1', 'b2'])
    with pytest.raises(TypeError):
        pack(df, a=['a1', 'a2'], b=['b1', 'b2'], extra_arg=None)
    with pytest.raises(ValueError):
        pack(df, a=['a1', 'a2'], names_sep=1)

# unpack ------------------------------------------------------------------

def test_non_df_cols_are_skipped():
    df = pd.DataFrame({'x': [1, 2], 'y': [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}]})
    out1 = unpack(df, columns='x')
    out2 = df.copy()
    pd.testing.assert_frame_equal(out1, out2)
    out3 = unpack(df, columns=list(df.columns))
    out4 = unpack(df, columns='y')
    pd.testing.assert_frame_equal(out3, out4)

def test_empty_columns_that_arent_dataframes_arent_unpacked():
    df = pd.DataFrame({'x': []})
    out = unpack(df, columns='x')
    pd.testing.assert_frame_equal(out, df)

def test_df_cols_are_directly_unpacked():
    df = pd.DataFrame({
        'x': [1, 2, 3],
        'y': [{'a': 1, 'b': 3}, {'a': 2, 'b': 2}, {'a': 3, 'b': 1}]
    })
    out = unpack(df, columns='y')
    assert list(out.columns) == ['x', 'a', 'b']
    pd.testing.assert_frame_equal(out[['a', 'b']], pd.DataFrame(df['y'].tolist()))

def test_can_unpack_zero_col_dataframe():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [{} for _ in range(3)]})
    out = unpack(df, columns='y')
    assert list(out.columns) == ['x']

def test_can_unpack_zero_row_dataframe():
    df = pd.DataFrame({'x': [], 'y': []})
    out = unpack(df, columns='y')
    pd.testing.assert_frame_equal(out, pd.DataFrame({'x': []}))

def test_can_choose_to_add_separator():
    df = pd.DataFrame({
        'x': [1],
        'y': [{'a': 2}],
        'z': [{'a': 3}]
    })
    out = unpack(df, columns=['y', 'z'], names_sep='_')
    assert list(out.columns) == ['x', 'y_a', 'z_a']

def test_unpacked_column_with_empty_dict():
    df = pd.DataFrame({'x': [{}]})
    out = unpack(df, columns='x')
    pd.testing.assert_frame_equal(out, pd.DataFrame(index=[0]))

def test_catches_inner_name_duplication():
    df = pd.DataFrame({
        'x': [{'a': 3, 'b': 4}],
        'y': [{'b': 5}],
        'z': [{'a': 6, 'b': 6}]
    })
    with pytest.raises(ValueError):
        unpack(df, columns=['x', 'y'])
    with pytest.raises(ValueError):
        unpack(df, columns=['x', 'y', 'z'])

def test_catches_outer_inner_name_duplication():
    df = pd.DataFrame({
        'a': [1],
        'b': [2],
        'c': [3],
        'd': [{'a': 4}],
        'e': [{'d': 5}],
        'f': [{'b': 6, 'c': 7, 'g': 8}]
    })
    with pytest.raises(ValueError):
        unpack(df, columns='d')
    with pytest.raises(ValueError):
        unpack(df, columns=['d', 'e', 'f'])

def test_duplication_error_not_triggered_on_same_name():
    df = pd.DataFrame({'x': [{'x': 1}]})
    out = unpack(df, columns='x')
    pd.testing.assert_frame_equal(out, pd.DataFrame({'x': [1]}))

def test_duplication_errors_with_names_sep_avoided():
    df1 = pd.DataFrame({
        'x': [1],
        'y': [{'x': 2}]
    })
    df2 = pd.DataFrame({
        'x': [{'a': 1}],
        'y': [{'a': 2}]
    })
    out1 = unpack(df1, columns='y', names_sep='_')
    pd.testing.assert_frame_equal(out1, pd.DataFrame({'x': [1], 'y_x': [2]}))
    out2 = unpack(df2, columns=['x', 'y'], names_sep='_')
    pd.testing.assert_frame_equal(out2, pd.DataFrame({'x_a': [1], 'y_a': [2]}))

def test_unpack_disallows_renaming():
    df = pd.DataFrame({'x': [{'a': 1}]})
    with pytest.raises(TypeError):
        unpack(df, columns={'y': 'x'})

def test_unpack_validates_its_inputs():
    df = pd.DataFrame({
        'x': [1, 2],
        'y': [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}]
    })
    with pytest.raises(TypeError):
        unpack(1)
    with pytest.raises(ValueError):
        unpack(df)
    with pytest.raises(ValueError):
        unpack(df, columns='y', names_sep=1)

# Run tests
if __name__ == '__main__':
    pytest.main([__file__])