import pandas as pd
import pytest

def uncount(df, weights, _remove=True, _id=None):
    """
    Expand dataframe based on weights.

    Parameters:
    df : pandas DataFrame
    weights : str (column name) or int
    _remove : bool, default True; if True, remove the weights column from result
    _id : str, default None; if provided, create an id column with this name

    Returns:
    Pandas DataFrame expanded according to weights
    """
    if isinstance(weights, (int, float)):
        w = pd.Series([int(weights)] * len(df))
    elif isinstance(weights, str):
        if weights not in df.columns:
            raise ValueError(f"Column '{weights}' not found in DataFrame.")
        w = df[weights]
    else:
        raise TypeError(f"Invalid type for weights: {type(weights)}")

    if not pd.api.types.is_numeric_dtype(w):
        raise ValueError("Weights must be numeric.")

    if (w < 0).any():
        raise ValueError("Weights must be non-negative integers.")

    if not all(w == w.astype(int)):
        raise ValueError("Weights must be integers.")

    w = w.astype(int)

    expanded_df = df.loc[df.index.repeat(w)].copy()

    if _id is not None:
        if not isinstance(_id, str) or not _id:
            raise ValueError("_id must be a non-empty string")
        group_sizes = w.values
        repeated_ids = [list(range(1, count + 1)) for count in group_sizes]
        expanded_df[_id] = [i for ids in repeated_ids for i in ids]

    if _remove and isinstance(weights, str):
        if weights in expanded_df.columns:
            del expanded_df[weights]

    return expanded_df.reset_index(drop=True)


def test_symbols_weights_are_dropped_in_output():
    df = pd.DataFrame({'x': [1], 'w': [1]})
    result = uncount(df, 'w')
    expected = pd.DataFrame({'x': [1]})
    pd.testing.assert_frame_equal(result, expected)

def test_can_request_to_preserve_symbols():
    df = pd.DataFrame({'x': [1], 'w': [1]})
    result = uncount(df, 'w', _remove=False)
    expected = df
    pd.testing.assert_frame_equal(result, expected)

def test_unique_identifiers_created_on_request():
    df = pd.DataFrame({'w': [1, 2, 3]})
    result = uncount(df, 'w', _id='id')
    expected = pd.DataFrame({'id': [1] + [1, 2] + [1, 2, 3]})
    pd.testing.assert_frame_equal(result, expected)

def test_expands_constants_and_expressions():
    df = pd.DataFrame({'x': [1], 'w': [2]})
    result1 = uncount(df, 2)
    result2 = uncount(df, 1 + 1)
    expected = pd.concat([df, df], ignore_index=True)
    pd.testing.assert_frame_equal(result1, expected)
    pd.testing.assert_frame_equal(result2, expected)

def test_works_with_groups():
    df = pd.DataFrame({'g': [1], 'x': [1], 'w': [1]})
    result = uncount(df, 'w')
    expected = df.drop(columns=['w'])
    pd.testing.assert_frame_equal(result, expected)

def test_must_evaluate_to_integer():
    df = pd.DataFrame({'x': [1], 'w': [0.5]})
    with pytest.raises(ValueError):
        uncount(df, 'w')

    df = pd.DataFrame({'x': [1]})
    with pytest.raises(ValueError):
        uncount(df, "W")

def test_works_with_zero_weights():
    df = pd.DataFrame({'x': [1, 2], 'w': [0, 1]})
    result = uncount(df, 'w')
    expected = pd.DataFrame({'x': [2]})
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

def test_validates_inputs():
    df = pd.DataFrame({'x': [1], 'y': ['a'], 'w': [-1]})

    with pytest.raises(ValueError):
        uncount(df, 'y')
    with pytest.raises(ValueError):
        uncount(df, 'w')
    with pytest.raises(TypeError):
        uncount(df, 'x', _remove=1)
    with pytest.raises(ValueError):
        uncount(df, 'x', _id="")