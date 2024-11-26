import pandas as pd
import numpy as np

def spread(data, key, value, fill=np.nan, convert=False, drop=True, sep=None):
    key_var = key
    value_var = value

    index_cols = data.columns.difference([key_var, value_var])
    if index_cols.empty:
        # Special case when there's no index columns
        data_pivoted = data.pivot(columns=key_var, values=value_var)
        data_pivoted = data_pivoted.reset_index(drop=True)
    else:
        data_pivoted = data.pivot_table(index=index_cols.tolist(), columns=key_var, values=value_var, dropna=drop, aggfunc='first')

    # Handle fill parameter
    if fill is not np.nan:
        data_pivoted = data_pivoted.fillna(fill)

    # Convert columns if convert is True
    if convert:
        data_pivoted = data_pivoted.apply(lambda x: pd.to_numeric(x, errors='ignore'))

    # Handle sep parameter
    if sep is not None:
        data_pivoted.columns = [str(key_var) + sep + str(col) for col in data_pivoted.columns]
    else:
        # Columns are just the keys
        data_pivoted.columns = data_pivoted.columns.astype(str)

    # Flatten MultiIndex columns if any
    if isinstance(data_pivoted.columns, pd.MultiIndex):
        data_pivoted.columns = ['_'.join([str(i) for i in col if str(i) != '']) for col in data_pivoted.columns.values]

    # Reset index to turn index into columns
    data_pivoted = data_pivoted.reset_index()
    return data_pivoted

def col_names(x, sep=None):
    names = x.iloc[:, 0].astype(str)

    if sep is None:
        if len(names) == 0:
            return []
        else:
            names = names.fillna('<NA>')
            return names.tolist()
    else:
        name = x.columns[0]
        return [str(name) + sep + str(n) for n in names]

def as_tibble_matrix(x):
    return pd.DataFrame(x)

def split_labels(df, id, drop=True):
    if df.empty:
        return df

    if drop:
        representative = pd.DataFrame().assign(id=id).drop_duplicates('id').index
        out = df.iloc[representative].reset_index(drop=True)
        return out
    else:
        unique_values = {col: df[col].unique() for col in df.columns}
        combinations = pd.MultiIndex.from_product(unique_values.values(), names=unique_values.keys())
        return pd.DataFrame(combinations.to_list(), columns=unique_values.keys())

def ulevels(x):
    if pd.api.types.is_categorical_dtype(x):
        categories = x.cat.categories
        return pd.Categorical(categories, categories=categories, ordered=x.cat.ordered)
    else:
        return np.sort(pd.unique(x))
