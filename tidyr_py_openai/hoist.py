import pandas as pd

def hoist(data,
          col,
          *args,
          remove=True,
          simplify=True,
          ptype=None,
          transform=None,
          **kwargs):
    # Hoist values out of list-columns
    #
    # hoist() allows you to selectively pull components of a list-column
    # into their own top-level columns.
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")

    if col not in data.columns:
        raise ValueError(f"col must be a column in data")

    pluckers = check_pluckers(*args, **kwargs)

    x = data[col]
    if not x.apply(lambda item: isinstance(item, (list, dict))).all():
        raise ValueError(f"data['{col}'] must be a list-like column")

    # Extract columns
    cols = {}
    for name, idx in pluckers.items():
        cols[name] = x.apply(lambda item: pluck(item, idx))

    # Simplify columns if needed
    cols = df_simplify(
        cols,
        ptype=ptype,
        transform=transform,
        simplify=simplify
    )

    # Insert new columns before the old column
    col_index = data.columns.get_loc(col)
    for offset, (new_col_name, new_col_values) in enumerate(cols.items()):
        data.insert(loc=col_index + offset, column=new_col_name, value=new_col_values)

    if remove:
        # Remove extracted components from x
        x_updated = []
        for item in x:
            # Reverse the pluckers
            for idx in reversed(list(pluckers.values())):
                item = strike(item, idx)
            x_updated.append(item)

        # If all items are empty, remove the column
        if all(is_empty(item) for item in x_updated):
            data.drop(columns=[col], inplace=True)
        else:
            data[col] = x_updated

    return data

def check_pluckers(*args, **kwargs):
    # Process the args into pluckers
    pluckers = {}

    # First process the keyword arguments
    for k, v in kwargs.items():
        pluckers[k] = v

    # Now process positional arguments
    for arg in args:
        if isinstance(arg, str):
            # Unnamed strings
            pluckers[arg] = [arg]
        else:
            raise ValueError("Invalid argument; positional arguments must be strings")

    # Ensure unique names
    if len(set(pluckers.keys())) != len(pluckers):
        raise ValueError("Column names must be unique in hoist()")

    # Standardize all pluckers to lists
    for k, v in pluckers.items():
        if not isinstance(v, list):
            pluckers[k] = [v]
    return pluckers

def pluck(item, idx):
    # Similar to purrr::pluck in R
    try:
        for key in idx:
            if isinstance(item, dict):
                item = item[key]
            elif isinstance(item, list):
                if isinstance(key, int):
                    item = item[key]
                else:
                    # Invalid key for list
                    return None
            else:
                # Cannot pluck further
                return None
        return item
    except (KeyError, IndexError, TypeError):
        return None

def df_simplify(cols, ptype=None, transform=None, simplify=True):
    # Simplify columns if possible, apply transforms and ptype
    for col_name, col_values in cols.items():
        if transform is not None:
            if isinstance(transform, dict):
                if col_name in transform:
                    func = transform[col_name]
                    col_values = col_values.apply(func)
            elif callable(transform):
                col_values = col_values.apply(transform)
        # Apply ptype if specified
        if ptype is not None:
            if isinstance(ptype, dict):
                if col_name in ptype:
                    expected_type = ptype[col_name]
                    col_values = col_values.astype(expected_type)
            else:
                # Single ptype for all columns
                col_values = col_values.astype(ptype)
        # Simplify if needed
        if isinstance(simplify, dict):
            simplify_col = simplify.get(col_name, True)
        else:
            simplify_col = simplify
        if simplify_col:
            # Attempt to simplify list-like columns
            if not col_values.apply(lambda v: isinstance(v, (list, dict))).any():
                col_values = pd.Series(col_values.tolist(), index=col_values.index)
        cols[col_name] = col_values
    return cols

def strike(item, indices):
    if not isinstance(indices, list):
        raise ValueError("indices must be a list")

    if not indices:
        return item

    index = indices[0]
    rest_indices = indices[1:]

    if isinstance(item, dict):
        if index in item:
            if rest_indices:
                item[index] = strike(item[index], rest_indices)
            else:
                del item[index]
    elif isinstance(item, list):
        if isinstance(index, int) and 0 <= index < len(item):
            if rest_indices:
                item[index] = strike(item[index], rest_indices)
            else:
                del item[index]
    return item

def is_empty(item):
    if item is None:
        return True
    if isinstance(item, (list, dict, set)) and len(item) == 0:
        return True
    return False