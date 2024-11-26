# unnest_auto -------------------------------------------------------------

import pandas as pd

def test_that(description, func):
    print(f"Test: {description}")
    func()

def expect_message(actual_message, expected_message):
    assert expected_message == actual_message, f"Expected message '{expected_message}', got '{actual_message}'"

def expect_equal(a, b):
    assert a == b, f"Expected {b}, got {a}"

def expect_named(df, names):
    assert list(df.columns) == names, f"Expected columns {names}, got {list(df.columns)}"

def unnest_longer(df, col_name):
    df = df.copy()
    col_data = df[col_name]
    if any(isinstance(x, dict) for x in col_data):
        # Handle elements that are dictionaries
        df['tmp'] = df[col_name].apply(
            lambda x: list(x.items()) if isinstance(x, dict) else [(None, x)]
        )
        df = df.explode('tmp')
        df[[f"{col_name}_id", col_name]] = pd.DataFrame(df['tmp'].tolist(), index=df.index)
        df = df.drop(columns=['tmp'])
        # Reset index
        df = df.reset_index(drop=True)
    else:
        # Simple case, explode the column
        df = df.explode(col_name).reset_index(drop=True)
    return df

def unnest_wider(df, col_name):
    col_data = df[col_name]
    new_cols = pd.DataFrame(col_data.tolist())
    new_df = pd.concat([df.drop(columns=[col_name]), new_cols], axis=1)
    return new_df

def unnest_auto(df, col_name):
    col_data = df[col_name]
    if all(isinstance(x, dict) for x in col_data):
        keys = [set(x.keys()) for x in col_data]
        if all(k == keys[0] for k in keys):
            message = "unnest_wider"
            result = unnest_wider(df, col_name)
            return result, message
    message = "unnest_longer"
    result = unnest_longer(df, col_name)
    return result, message

test_that("unnamed becomes longer", lambda: (
    (lambda df:
        (lambda out, message:
            (
                expect_message(message, "unnest_longer"),
                expect_equal(list(out['y']), [1, 2, 3])
            )
        )(*unnest_auto(df, 'y'))
    )(pd.DataFrame({'x': [1, 2], 'y': [1, [2, 3]]}))
))

test_that("common name becomes wider", lambda: (
    (lambda df:
        (lambda out, message:
            (
                expect_message(message, "unnest_wider"),
                expect_named(out, ['x', 'a'])
            )
        )(*unnest_auto(df, 'y'))
    )(pd.DataFrame({
        'x': [1, 2],
        'y': [{'a': 1}, {'a': 2}]
    }))
))

test_that("no common name falls back to longer with index", lambda: (
    (lambda df:
        (lambda out, message:
            (
                expect_message(message, "unnest_longer"),
                expect_named(out, ['x', 'y_id', 'y'])
            )
        )(*unnest_auto(df, 'y'))
    )(pd.DataFrame({
        'x': [1, 2],
        'y': [{'a': 1}, {'b': 2}]
    }))
))

test_that("mix of named and unnamed becomes longer", lambda: (
    (lambda df:
        (lambda out, message:
            (
                expect_message(message, "unnest_longer"),
                expect_named(out, ['x', 'y'])
            )
        )(*unnest_auto(df, 'y'))
    )(pd.DataFrame({
        'x': [1, 2],
        'y': [{'a': 1}, 2]
    }))
))

# https://github.com/tidyverse/tidyr/issues/959
test_that("works with an input that has column named `col`", lambda: (
    (lambda df:
        (lambda out, message:
            (
                expect_message(message, "unnest_wider"),
                expect_named(out, ['col', 'x', 'y'])
            )
        )(*unnest_auto(df, 'list_col'))
    )(pd.DataFrame({
        'col': [1, 1],
        'list_col': [{'x': 'a', 'y': 'b'}, {'x': 'c', 'y': 'd'}]
    }))
))