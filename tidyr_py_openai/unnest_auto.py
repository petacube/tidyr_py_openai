def unnest_auto(data, col):
    # Ensure col is specified
    if col is None:
        raise ValueError("col is required")

    x = data[col]
    dir = guess_dir(x, col)

    if dir == 'longer':
        return unnest_longer(data, col, indices_include=False)
    elif dir == 'longer_idx':
        return unnest_longer(data, col, indices_include=True)
    elif dir == 'wider':
        return unnest_wider(data, col, names_repair='unique')

def guess_dir(x, col):
    # x is a pandas Series
    # col is the column name
    names_list = []
    for elem in x:
        if isinstance(elem, dict):
            names_list.append(list(elem.keys()))
        else:
            names_list.append(None)  # Names are None

    is_null = set(name is None for name in names_list)

    if is_null == {True}:
        # all unnamed
        code = f"unnest_longer({col}, indices_include=False)"
        reason = "no element has names"
        out = "longer"
    elif is_null == {False}:
        # all named
        common_names = set(names_list[0])
        for names in names_list[1:]:
            common_names.intersection_update(names)

        n_common = len(common_names)
        if n_common == 0:
            code = f"unnest_longer({col}, indices_include=True)"
            reason = "elements are named, but have no names in common"
            out = "longer_idx"
        else:
            code = f"unnest_wider({col})"
            reason = f"elements have {n_common} names in common"
            out = "wider"
    else:
        # mix of named and unnamed elements
        code = f"unnest_longer({col}, indices_include=False)"
        reason = "mix of named and unnamed elements"
        out = "longer"

    print(f"Using `{code}`; {reason}")

    return out

def unnest_longer(data, col, indices_include=False):
    # data: pandas DataFrame
    # col: column name
    # indices_include: if True, include indices (from the list elements) in the output
    import pandas as pd

    data_long = data.copy()
    data_long[col] = data_long[col].apply(lambda x: [x] if not isinstance(x, list) else x)

    if indices_include:
        data_long = data_long.explode(col)
        data_long['index'] = data_long.groupby(level=0).cumcount()
    else:
        data_long = data_long.explode(col)

    data_long = data_long.reset_index(drop=True)

    return data_long

def unnest_wider(data, col, names_repair='unique'):
    # data: pandas DataFrame
    # col: column name to unnest
    # names_repair: 'unique' means make sure that the new columns have unique names
    import pandas as pd

    data_wide = data.copy()
    data_wide[col] = data_wide[col].apply(lambda x: x if isinstance(x, dict) else {})

    new_cols = pd.json_normalize(data_wide[col])

    if names_repair == 'unique':
        # Ensure unique column names
        cols = list(new_cols.columns)
        existing_cols = set(data_wide.columns)
        col_counts = {}
        for idx, col_name in enumerate(cols):
            if col_name in existing_cols:
                count = col_counts.get(col_name, 1)
                new_col_name = f"{col_name}_{count}"
                while new_col_name in existing_cols:
                    count += 1
                    new_col_name = f"{col_name}_{count}"
                col_counts[col_name] = count + 1
                cols[idx] = new_col_name
                existing_cols.add(new_col_name)
            else:
                existing_cols.add(col_name)
        new_cols.columns = cols

    data_wide = pd.concat([data_wide.drop(columns=[col]), new_cols], axis=1)

    return data_wide