import pandas as pd
import numpy as np

def drop_na(data, *args,allow_empty_params=False):
    """
    Drop rows containing missing values

    `drop_na()` drops rows where any column specified by `*args` contains a
    missing value.

    Another way to interpret `drop_na()` is that it only keeps the "complete"
    rows (where no rows contain missing values).

    Args:
        data: A DataFrame.
        *args: Columns to inspect for missing values. If empty, all columns are used.
               Each argument can be a column name (str) or a list of column names.

    Examples:
        df = pd.DataFrame({'x': [1, 2, np.nan], 'y': ['a', np.nan, 'b']})
        drop_na(df)
        drop_na(df, 'x')

        vars = ['y']
        drop_na(df, 'x', *vars)
    """
    # Ensure no named arguments are supplied
    if any(not (isinstance(arg, str) or
                isinstance(arg, list))
           for arg in args):
        raise ValueError("All arguments after data must be column names or lists of column names")
    # manually added by SS - check for empty list as param
    for arg in args:
        if not arg:
            if not allow_empty_params:
                raise ValueError("You cant pass empty list as parameter")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    # Flatten list of args to get all columns
    if not args:
        # Use all columns if no columns are specified
        cols = data.columns.tolist()
    else:
        cols = []
        for arg in args:
            if isinstance(arg, str):
                cols.append(arg)
            elif isinstance(arg, list):
                cols.extend(arg)
            else:
                raise ValueError("Arguments must be column names or lists of column names")
        # Remove duplicates
        cols = list(dict.fromkeys(cols))
    # Verify that columns exist in data
    missing_cols = [col for col in cols if col not in data.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in data: {missing_cols}")
    # Boolean Series where rows have no NA in specified columns
    loc = data[cols].notna().all(axis=1)
    out = data.loc[loc]
    return out
