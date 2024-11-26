import warnings
import pandas as pd

# Deprecated SE versions of main verbs
#
# tidyr used to offer twin versions of each verb suffixed with an
# underscore. These versions had standard evaluation (SE) semantics:
# rather than taking arguments by code, like NSE verbs, they took
# arguments by value. Their purpose was to make it possible to
# program with tidyr. However, tidyr now uses tidy evaluation
# semantics. NSE verbs still capture their arguments, but you can now
# unquote parts of these arguments. This offers full programmability
# with NSE verbs. Thus, the underscored versions are now superfluous.
#
# Unquoting triggers immediate evaluation of its operand and inlines
# the result within the captured expression. This result can be a
# value or an expression to be evaluated later with the rest of the
# argument. See the dplyr programming vignette for more information.

def complete_(data, cols, fill=None, *args, **kwargs):
    warnings.warn("1.0.0", DeprecationWarning)
    if isinstance(data, pd.DataFrame):
        return complete__data_frame(data, cols, fill=fill, *args, **kwargs)
    else:
        raise TypeError("Unsupported data type for complete_")

def complete__data_frame(data, cols, fill=None, *args, **kwargs):
    # Placeholder for 'complete' function
    # The actual implementation would depend on the specifics of the complete function
    return data  # Replace with the actual implementation

def drop_na_(data, vars):
    warnings.warn("1.0.0", DeprecationWarning)
    if isinstance(data, pd.DataFrame):
        return drop_na__data_frame(data, vars)
    else:
        raise TypeError("Unsupported data type for drop_na_")

def drop_na__data_frame(data, vars):
    return data.dropna(subset=vars)

def expand_(data, dots, *args, **kwargs):
    warnings.warn("1.0.0", DeprecationWarning)
    if isinstance(data, pd.DataFrame):
        return expand__data_frame(data, dots, *args, **kwargs)
    else:
        raise TypeError("Unsupported data type for expand_")

def expand__data_frame(data, dots, *args, **kwargs):
    # Placeholder for 'expand' function
    return data  # Replace with the actual implementation

def crossing_(x):
    warnings.warn("1.0.0", DeprecationWarning)
    # Placeholder for 'crossing' function
    return x  # Replace with the actual implementation

def nesting_(x):
    warnings.warn("1.2.0", DeprecationWarning)
    # Placeholder for 'nesting' function
    return x  # Replace with the actual implementation

def extract_(data, col, into, regex="([\\w]+)", remove=True, convert=False, *args, **kwargs):
    warnings.warn("1.0.0", DeprecationWarning)
    if isinstance(data, pd.DataFrame):
        return extract__data_frame(data, col, into, regex=regex, remove=remove, convert=convert, *args, **kwargs)
    else:
        raise TypeError("Unsupported data type for extract_")

def extract__data_frame(data, col, into, regex="([\\w]+)", remove=True, convert=False, *args, **kwargs):
    extracted = data[col].str.extract(regex)
    extracted.columns = into
    if convert:
        extracted = extracted.apply(pd.to_numeric, errors='ignore')
    if remove:
        data = data.drop(columns=[col])
    data = pd.concat([data, extracted], axis=1)
    return data

def fill_(data, fill_cols, direction="down"):
    warnings.warn("1.2.0", DeprecationWarning)
    if isinstance(data, pd.DataFrame):
        return fill__data_frame(data, fill_cols, direction=direction)
    else:
        raise TypeError("Unsupported data type for fill_")

def fill__data_frame(data, fill_cols, direction="down"):
    method = {'down': 'ffill', 'up': 'bfill'}.get(direction)
    if method is None:
        raise ValueError("Invalid direction")
    data[fill_cols] = data[fill_cols].fillna(method=method)
    return data

def gather_(data, key_col, value_col, gather_cols, na_rm=False, convert=False, factor_key=False):
    warnings.warn("1.2.0", DeprecationWarning)
    if isinstance(data, pd.DataFrame):
        return gather__data_frame(data, key_col, value_col, gather_cols, na_rm=na_rm, convert=convert, factor_key=factor_key)
    else:
        raise TypeError("Unsupported data type for gather_")

def gather__data_frame(data, key_col, value_col, gather_cols, na_rm=False, convert=False, factor_key=False):
    id_vars = [col for col in data.columns if col not in gather_cols]
    melted = pd.melt(data, id_vars=id_vars, value_vars=gather_cols, var_name=key_col, value_name=value_col)
    if na_rm:
        melted = melted.dropna(subset=[value_col])
    return melted

def nest_(*args, **kwargs):
    warnings.warn("1.0.0", DeprecationWarning)
    raise NotImplementedError("nest_() is deprecated and cannot be used")

def separate_rows_(data, cols, sep="[^\\w.]+", convert=False):
    warnings.warn("1.2.0", DeprecationWarning)
    if isinstance(data, pd.DataFrame):
        return separate_rows__data_frame(data, cols, sep=sep, convert=convert)
    else:
        raise TypeError("Unsupported data type for separate_rows_")

def separate_rows__data_frame(data, cols, sep="[^\\w.]+", convert=False):
    import numpy as np
    idx = data.index.repeat(data[cols[0]].str.split(sep).apply(len))
    df = data.loc[idx].reset_index(drop=True)
    for col in cols:
        df[col] = np.concatenate(data[col].str.split(sep).values)
        if convert:
            df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

def separate_(data, col, into, sep="[^\\w]+", remove=True, convert=False, extra="warn", fill="warn", *args, **kwargs):
    warnings.warn("1.2.0", DeprecationWarning)
    if isinstance(data, pd.DataFrame):
        return separate__data_frame(data, col, into, sep=sep, remove=remove, convert=convert, extra=extra, fill=fill, *args, **kwargs)
    else:
        raise TypeError("Unsupported data type for separate_")

def separate__data_frame(data, col, into, sep="[^\\w]+", remove=True, convert=False, extra="warn", fill="warn", *args, **kwargs):
    new_cols = data[col].str.split(sep, expand=True)
    new_cols = new_cols.iloc[:, :len(into)]
    new_cols.columns = into
    if remove:
        data = data.drop(columns=[col])
    data = pd.concat([data, new_cols], axis=1)
    if convert:
        data = data.apply(pd.to_numeric, errors='ignore')
    return data

def spread_(data, key_col, value_col, fill=None, convert=False, drop=True, sep=None):
    warnings.warn("1.2.0", DeprecationWarning)
    if isinstance(data, pd.DataFrame):
        return spread__data_frame(data, key_col, value_col, fill=fill, convert=convert, drop=drop, sep=sep)
    else:
        raise TypeError("Unsupported data type for spread_")

def spread__data_frame(data, key_col, value_col, fill=None, convert=False, drop=True, sep=None):
    spread_df = data.pivot(index=[col for col in data.columns if col not in [key_col, value_col]], columns=key_col, values=value_col)
    if fill is not None:
        spread_df = spread_df.fillna(fill)
    if convert:
        spread_df = spread_df.apply(pd.to_numeric, errors='ignore')
    spread_df = spread_df.reset_index()
    return spread_df

def unite_(data, col, from_cols, sep="_", remove=True):
    warnings.warn("1.2.0", DeprecationWarning)
    if isinstance(data, pd.DataFrame):
        return unite__data_frame(data, col, from_cols, sep=sep, remove=remove)
    else:
        raise TypeError("Unsupported data type for unite_")

def unite__data_frame(data, col, from_cols, sep="_", remove=True):
    data[col] = data[from_cols].astype(str).agg(sep.join, axis=1)
    if remove:
        data = data.drop(columns=from_cols)
    return data

def unnest_(*args, **kwargs):
    warnings.warn("1.0.0", DeprecationWarning)
    raise NotImplementedError("unnest_() is deprecated and cannot be used")