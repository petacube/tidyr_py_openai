import pandas as pd

def check_data_frame(df, call=None):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`data` must be a DataFrame.")

def check_string(s, allow_null=False, call=None):
    if not (isinstance(s, str) or (allow_null and s is None)):
        raise TypeError("`names_sep` must be a string or None.")

def strip_names(df, base, names_sep):
    prefix = base + names_sep
    new_columns = [col[len(prefix):] if col.startswith(prefix) else col for col in df.columns]
    df.columns = new_columns
    return df

def pack(data, names_sep=None, **kwargs):
    check_data_frame(data)
    if any(not name for name in kwargs.keys()):
        raise TypeError("All elements of `...` must be named.")
    check_string(names_sep, allow_null=True)

    # Process the kwargs into cols - dictionary mapping new column names to lists of column names
    cols = {}
    for new_col_name, col_names in kwargs.items():
        if not isinstance(col_names, (list, tuple)):
            raise TypeError(f"Columns to pack must be specified as a list or tuple of column names for `{new_col_name}`.")
        if any(col not in data.columns for col in col_names):
            missing = [col for col in col_names if col not in data.columns]
            raise KeyError(f"Columns {missing} not found in data.")
        cols[new_col_name] = col_names

    # Get the list of columns to remain unpacked
    all_packed_cols = [col for cols_list in cols.values() for col in cols_list]
    unpacked_cols = [col for col in data.columns if col not in all_packed_cols]

    # Prepare the unpacked DataFrame
    unpacked = data[unpacked_cols].copy()

    # Prepare the packed columns
    for new_col_name, col_names in cols.items():
        packed_data = data[col_names].copy()
        if names_sep is not None:
            packed_data = strip_names(packed_data, base=new_col_name, names_sep=names_sep)
        # The packed column is a Series where each row is a dictionary of the packed values
        packed_series = packed_data.apply(lambda row: row.to_dict(), axis=1).rename(new_col_name)
        unpacked[new_col_name] = packed_series

    return unpacked

def rename_with_names_sep(df, outer, names_sep):
    df = df.copy()
    df.columns = [f"{outer}{names_sep}{col}" for col in df.columns]
    return df

def unpack(data, cols, names_sep=None, names_repair='check_unique'):
    check_data_frame(data)
    if cols is None:
        raise ValueError("`cols` is required.")
    check_string(names_sep, allow_null=True)

    if isinstance(cols, str):
        cols = [cols]
    elif not isinstance(cols, (list, tuple)):
        raise TypeError("`cols` must be a string or list of strings.")

    # Make a copy of data to avoid modifying original DataFrame
    out = data.copy()

    # For each column in cols
    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Column `{col}` not found in data.")

        if out[col].empty:
            continue  # Skip empty columns

        # Check if the column is a Series of dicts
        if out[col].apply(lambda x: isinstance(x, dict)).all():
            expanded_cols = pd.DataFrame(out[col].tolist(), index=out.index)
        else:
            continue  # Do not unpack columns that are not Series of dicts

        # If names_sep is provided, combine outer and inner names
        if names_sep is not None and not expanded_cols.empty:
            expanded_cols = rename_with_names_sep(expanded_cols, col, names_sep)

        # Check for duplicate column names
        duplicate_names = set(out.columns) & set(expanded_cols.columns)
        duplicate_names.discard(col)  # Exclude the original column being unpacked
        if duplicate_names and names_repair == 'check_unique':
            raise ValueError(f"Duplicate column names after unpacking: {duplicate_names}")

        # Remove the original column
        out = out.drop(columns=[col])

        # Concatenate the expanded columns
        out = pd.concat([out, expanded_cols], axis=1)

    return out