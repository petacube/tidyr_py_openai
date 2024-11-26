import pandas as pd
import pytest
from pandas.api.types import is_list_like
from pandas.api.types import is_scalar

def hoist(df, column, *args, .simplify=True, .ptype=None, .transform=None, .remove=True):
    new_cols = {}
    col_data = df[column]
    for arg in args:
        new_cols[arg] = col_data.apply(lambda x: x[arg] if isinstance(x, dict) and arg in x else None)
    for key, plucker in kwargs.items():
        if isinstance(plucker, list):
            new_cols[key] = col_data.apply(lambda x: pluck_deep(x, plucker))
        else:
            new_cols[key] = col_data.apply(lambda x: x[plucker - 1] if isinstance(plucker, int) and isinstance(x, list) else x.get(plucker) if isinstance(x, dict) else None)
    if .ptype:
        for col, ptype in .ptype.items():
            new_cols[col] = new_cols[col].astype(ptype)
    if .transform:
        for col, func in .transform.items():
            new_cols[col] = new_cols[col].apply(func)
    if .simplify:
        for col in new_cols:
            if new_cols[col].apply(lambda x: is_list_like(x) and len(x) == 1).all():
                new_cols[col] = new_cols[col].apply(lambda x: x[0] if is_list_like(x) else x)
    if .remove:
        df = df.drop(columns=[column])
    return df.assign(**new_cols)

def pluck_deep(x, plucker):
    for key in plucker:
        if isinstance(x, dict) and key in x:
            x = x[key]
        elif isinstance(x, list) and isinstance(key, int) and key - 1 < len(x):
            x = x[key - 1]
        else:
            return None
    return x

def test_hoist_extracts_named_elements():
    df = pd.DataFrame({'x': [{'1': 1, 'b': 'b'}]})

    out = hoist(df, 'x', a='1', b='b')
    expected = pd.DataFrame({'a': [1], 'b': ['b']})
    pd.testing.assert_frame_equal(out, expected)

    out = hoist(df, 'x', a='1', b='b', .simplify=False)
    expected = pd.DataFrame({'a': [[1]], 'b': [['b']]})
    pd.testing.assert_frame_equal(out, expected)

def test_can_hoist_named_non_list_elements_at_the_deepest_level():
    df = pd.DataFrame({'x': [{'a': {'aa': 1, 'bb': 2}}]})
    out = hoist(df, 'x', bb=['a', 'bb'])
    expected = pd.DataFrame({'bb': [2]})
    pd.testing.assert_frame_equal(out, expected)

def test_can_check_transform_values():
    df = pd.DataFrame({'x': [{'a': 1}, {'a': 'a'}]})

    with pytest.raises(TypeError):
        hoist(df, 'x', a='a', .ptype={'a': str})

    out = hoist(df, 'x', a='a', .transform={'a': str})
    expected = pd.DataFrame({'a': ['1', 'a']})
    pd.testing.assert_frame_equal(out, expected)

def test_nested_lists_generate_a_cast_error_if_they_cant_be_cast_to_the_ptype():
    df = pd.DataFrame({'x': [{'b': [1]}]})

    with pytest.raises(TypeError):
        hoist(df, 'x', 'b', .ptype={'b': float})

def test_non_vectors_generate_a_cast_error_if_a_ptype_is_supplied():
    df = pd.DataFrame({'x': [{'b': 'a'}]})

    with pytest.raises(TypeError):
        hoist(df, 'x', 'b', .ptype={'b': int})

def test_a_ptype_generates_a_list_of_ptype_if_the_col_cant_be_simplified():
    df = pd.DataFrame({'x': [{'a': [1, 2]}, {'a': 1}, {'a': 1}]})
    ptype = {'a': int}

    out = hoist(df, 'x', 'a', .ptype=ptype)
    expected = pd.DataFrame({'a': [[1, 2], 1, 1]})
    pd.testing.assert_frame_equal(out, expected)

def test_doesnt_simplify_uneven_lengths():
    df = pd.DataFrame({'x': [{'a': 1}, {'a': [2, 3]}]})

    out = hoist(df, 'x', a='a')
    expected = pd.DataFrame({'a': [1, [2, 3]]})
    pd.testing.assert_frame_equal(out, expected)

def test_doesnt_simplify_lists_of_lists():
    df = pd.DataFrame({'x': [{'a': [1]}, {'a': [2]}]})

    out = hoist(df, 'x', a='a')
    expected = pd.DataFrame({'a': [[1], [2]]})
    pd.testing.assert_frame_equal(out, expected)

def test_doesnt_simplify_non_vectors():
    df = pd.DataFrame({'x': [{'a': 'a'}, {'a': 'b'}]})

    out = hoist(df, 'x', a='a')
    expected = pd.DataFrame({'a': ['a', 'b']})
    pd.testing.assert_frame_equal(out, expected)

def test_can_hoist_out_scalars():
    from sklearn.linear_model import LinearRegression

    df = pd.DataFrame({
        'x': [1, 2],
        'y': [{'mod': LinearRegression().fit([[1]], [1])}, {'mod': LinearRegression().fit([[1]], [1])}]
    })
    out = hoist(df, 'y', 'mod')
    expected = pd.DataFrame({'x': [1, 2], 'mod': df['y'].apply(lambda x: x['mod'])})
    pd.testing.assert_frame_equal(out, expected)

def test_input_validation_catches_problems():
    df = pd.DataFrame({'x': [{'1': 1, 'b': 'b'}], 'y': [1]})

    with pytest.raises(ValueError):
        hoist(df, 'y')
    with pytest.raises(ValueError):
        hoist(df, 'x', 1)
    with pytest.raises(ValueError):
        hoist(df, 'x', a='a', a='b')

def test_string_pluckers_are_automatically_named():
    out = hoist(pd.DataFrame(), 'x', y='x', z=1)
    assert list(out.columns) == ['y', 'z']

def test_cant_hoist_from_a_data_frame_column():
    df = pd.DataFrame({'a': pd.DataFrame({'x': [1]})})

    with pytest.raises(TypeError):
        hoist(df, 'a', xx=1)

def test_can_hoist_without_any_pluckers():
    df = pd.DataFrame({'a': [1]})
    out = hoist(df, 'a')
    expected = df
    pd.testing.assert_frame_equal(out, expected)

def test_can_use_a_character_vector_for_deep_hoisting():
    df = pd.DataFrame({'x': [{'b': {'a': 1}}]})
    out = hoist(df, 'x', ba=['b', 'a'])
    expected = pd.DataFrame({'ba': [1]})
    pd.testing.assert_frame_equal(out, expected)

def test_can_use_a_numeric_vector_for_deep_hoisting():
    df = pd.DataFrame({'x': [{'b': {'a': 1, 'b': 2}}]})
    out = hoist(df, 'x', bb=[1, 2])
    expected = pd.DataFrame({'bb': [2]})
    pd.testing.assert_frame_equal(out, expected)

def test_can_maintain_type_stability_with_empty_elements():
    df = pd.DataFrame({
        'col': [{'a': []}, {'a': []}]
    })

    out = hoist(df, 'col', 'a')
    expected = pd.DataFrame({'a': [None, None]})
    pd.testing.assert_frame_equal(out, expected)

def test_can_hoist_out_a_rcrd_style_column():
    df = pd.DataFrame({'a': [{'x': 1, 'y': 2}, {'x': 1, 'y': 2}]})

    out = hoist(df, 'a', 'x')
    expected = pd.DataFrame({'x': [1, 1]})
    pd.testing.assert_frame_equal(out, expected)

def test_hoist_validates_its_inputs():
    df = pd.DataFrame({'a': [1]})

    with pytest.raises(TypeError):
        hoist(1)
    with pytest.raises(ValueError):
        hoist(df)
    with pytest.raises(TypeError):
        hoist(df, 'a', .remove=1)
    with pytest.raises(TypeError):
        hoist(df, 'a', .ptype=1)
    with pytest.raises(TypeError):
        hoist(df, 'a', .transform=1)
    with pytest.raises(TypeError):
        hoist(df, 'a', .simplify=1)

def test_hoist_can_simplify_on_a_per_column_basis():
    df = pd.DataFrame({
        'x': [{'a': 1, 'b': 1}, {'a': 2, 'b': 2}]
    })

    out = hoist(df, 'x', a='a', b='b', .simplify={'a': False})
    expected = pd.DataFrame({'a': [1, 2], 'b': [1, 2]})
    pd.testing.assert_frame_equal(out, expected)

def test_hoist_retrieves_first_of_duplicated_names_and_leaves_the_rest_alone():
    elt = {'x': 1, 'y': 2, 'x': 3, 'z': 2}
    df = pd.DataFrame({'col': [elt]})

    out = hoist(df, 'col', x='x')
    expected = pd.DataFrame({'x': [3], 'col': [{'y': 2, 'z': 2}]})
    pd.testing.assert_frame_equal(out, expected)

    out = hoist(df, 'col', y='y')
    expected = pd.DataFrame({'y': [2], 'col': [{'x': 3, 'z': 2}]})
    pd.testing.assert_frame_equal(out, expected)

def test_hoist_retains_grouped_data_frame_class():
    df = pd.DataFrame({
        'g': ['x', 'x', 'z'],
        'data': [{'a': [1, 2]}, {'a': [2, 3]}, {'a': [3, 4]}]
    })
    grouped = df.groupby('g')

    out = hoist(grouped, 'data', 'a')
    expected = df.assign(a=df['data'].apply(lambda x: x['a'])).groupby('g')
    pd.testing.assert_frame_equal(out, expected)

def test_hoist_retains_bare_data_frame_class():
    df = pd.DataFrame({
        'data': [{'a': [1, 2]}, {'a': [2, 3]}, {'a': [3, 4]}]
    })

    out = hoist(df, 'data', 'a')
    expected = pd.DataFrame({'a': [[1, 2], [2, 3], [3, 4]]})
    pd.testing.assert_frame_equal(out, expected)

def test_known_bug_hoist_doesnt_strike_after_each_pluck():
    elt = {'x': 1, 'x': 3, 'z': 2}
    df = pd.DataFrame({'col': [elt]})

    out = hoist(df, 'col', x1='x', x2='x')
    expected = pd.DataFrame({'x1': [3], 'x2': [3], 'col': [{'z': 2}]})
    pd.testing.assert_frame_equal(out, expected)

def strike(x, idx_list):
    if not idx_list:
        return x
    idx = idx_list[0]
    if isinstance(x, dict) and idx in x:
        x = x.copy()
        del x[idx]
    elif isinstance(x, list) and isinstance(idx, int) and idx - 1 < len(x):
        x = x[:idx - 1] + x[idx:]
    return x

def test_strike_can_remove_using_a_list():
    x = {'a': {}, 'b': {'a': 1, 'b': 2}, 'c': 'c'}

    assert strike(x, [1]) == {'b': {'a': 1, 'b': 2}, 'c': 'c'}
    assert strike(x, ['a']) == {'b': {'a': 1, 'b': 2}, 'c': 'c'}

    deep = strike(x, ['b', 2])
    assert deep == {'a': {}, 'b': {'a': 1}, 'c': 'c'}

def test_strike_returns_input_if_idx_not_present():
    x = {'a': {}, 'b': {'a': 1, 'b': 2}, 'c': 'c'}

    assert strike(x, [4]) == x
    assert strike(x, ['d']) == x
    assert strike(x, ['b', 3]) == x
    assert strike(x, ['d', 3]) == x
    assert strike(x, ['b', 'c']) == x
    assert strike(x, [3, 'b']) == x
    assert strike(x, [4, 'b']) == x

def test_ignores_weird_inputs():
    x = {'a': {}, 'b': {'a': 1, 'b': 2}, 'c': 'c'}

    assert strike(x, []) == x
    assert strike(x, [sum, sum]) == x