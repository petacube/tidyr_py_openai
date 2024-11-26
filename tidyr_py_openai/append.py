def df_append(x, y, after=None, remove=False):
    """
    Append new columns (`y`) to an existing data frame (`x`).

    If columns are duplicated between `x` and `y`, then `y` columns are
    silently preferred.

    Parameters
    ----------
    x : pandas.DataFrame
        A data frame.
    y : dict
        A named dictionary of columns to append. Each column must be the same size as `x`.
    after : int or str or None, optional
        One of:
        - `None` to place `y` at the end.
        - A single column name from `x` to place `y` after.
        - A single integer position (including `0`) to place `y` after.
    remove : bool, default False
        Whether or not to remove the column corresponding to `after` from `x`.

    Returns
    -------
    pandas.DataFrame
        A data frame containing the columns from `x` and any appended columns from `y`.
        The type of `x` is not maintained.
    """
    size = len(x)
    row_names = x.index

    x_names = list(x.columns)
    y_names = list(y.keys())

    n = len(x_names)

    if after is None:
        after = n
    elif isinstance(after, str):
        if after in x_names:
            after = x_names.index(after)
        else:
            raise ValueError(f"Column {after} not found in `x`.")
    elif isinstance(after, int):
        pass
    else:
        raise ValueError("`after` must be an integer, a column name from `x`, or `None`.")

    if not (0 <= after <= n):
        raise ValueError(f"`after` must be between 0 and {n}, inclusive.")

    if remove:
        if after <= 1:
            lhs_columns = []
        else:
            lhs_columns = x_names[:after - 1]
    else:
        if after < 1:
            lhs_columns = []
        else:
            lhs_columns = x_names[:after]

    rhs_columns = x_names[after:]

    # Prefer `y` if names are duplicated
    lhs_columns = [col for col in lhs_columns if col not in y_names]
    rhs_columns = [col for col in rhs_columns if col not in y_names]

    x_lhs = x[lhs_columns]
    x_rhs = x[rhs_columns]

    y_df = pd.DataFrame(y, index=row_names)

    out = pd.concat([x_lhs, y_df, x_rhs], axis=1)

    return out