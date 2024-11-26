import pandas as pd
import numpy as np

def reconstruct_tibble(input, output, ungrouped_vars=[]):
    # Reconstruct tibble based on input and output DataFrames
    # Grouping is not handled in this simplified version
    return output

def seq_nrow(x):
    # Returns a sequence from 1 to the number of rows in x
    return list(range(1, x.shape[0] + 1))

def seq_ncol(x):
    # Returns a sequence from 1 to the number of columns in x
    return list(range(1, x.shape[1] + 1))

def last(x):
    # Returns the last element of x
    return x[-1]

def make_unique(names, sep=''):
    # Makes names unique by appending suffixes
    counts = {}
    result = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            result.append(name)
        else:
            counts[name] += 1
            new_name = f"{name}{sep}{counts[name]}"
            while new_name in counts:
                counts[name] += 1
                new_name = f"{name}{sep}{counts[name]}"
            counts[new_name] = 0
            result.append(new_name)
    return result

def tidyr_legacy(nms, prefix="V", sep=""):
    # Ensures all column names are unique using legacy approach
    if len(nms) == 0:
        return []
    blank = [name == "" for name in nms]
    nms_non_blank = [nms[i] for i in range(len(nms)) if not blank[i]]
    nms_non_blank_unique = make_unique(nms_non_blank, sep=sep)
    nms_new = nms.copy()
    idx = 0
    for i in range(len(nms)):
        if not blank[i]:
            nms_new[i] = nms_non_blank_unique[idx]
            idx +=1
    num_blanks = sum(blank)
    possible_new_nms = [f"{prefix}{sep}{i+1}" for i in range(len(nms))]
    existing_names = set(nms_new)
    new_nms_candidates = [name for name in possible_new_nms if name not in existing_names]
    blank_indices = [i for i, b in enumerate(blank) if b]
    for idx_blank, idx_new_name in zip(blank_indices, range(len(new_nms_candidates))):
        nms_new[idx_blank] = new_nms_candidates[idx_new_name]
    return nms_new

def tidyr_col_modify(data, cols):
    # Modify DataFrame by adding or replacing columns from cols
    check_data_frame(data)
    if not isinstance(cols, dict):
        raise Exception("`cols` must be a dict.")
    data_dict = tidyr_new_list(data)
    for name, col in cols.items():
        data_dict[name] = col
    data = pd.DataFrame(data_dict)
    return data

def tidyr_new_list(x):
    # Returns a new list with only names as attributes
    if isinstance(x, pd.DataFrame):
        x_new = {col: x[col].tolist() for col in x.columns}
    elif isinstance(x, dict):
        x_new = x.copy()
    else:
        raise Exception("`x` must be a DataFrame or dict.")
    return x_new

def list_replace_null(x, sizes, ptype=None, size=1):
    # Replace None elements in x with default values
    if any(item is None for item in x):
        null_indices = [i for i, item in enumerate(x) if item is None]
        replacement = [np.nan] * size if ptype is None else [ptype] * size
        for i in null_indices:
            x[i] = replacement
            if size != 0:
                sizes[i] = size
    return {'x': x, 'sizes': sizes}

def list_replace_empty_typed(x, sizes, ptype=None, size=1):
    # Replace empty typed elements in x with default values
    if any(len(item) == 0 for item in x):
        empty_indices = [i for i, item in enumerate(x) if len(item) == 0]
        for i in empty_indices:
            x[i] = [ptype] * size if ptype is not None else [x[i]] * size
            sizes[i] = size
    return {'x': x, 'sizes': sizes}

def list_all_vectors2(x):
    # Check if all elements in x are vectors, excluding None
    x = [item for item in x if item is not None]
    return all(isinstance(item, (list, np.ndarray, pd.Series)) for item in x)

def list_of_ptype(x):
    # Return the prototype (ptype) attribute of x
    ptype = getattr(x, 'ptype', None)
    return ptype

def apply_names_sep(outer, inner, names_sep):
    # Apply names_sep by concatenating outer and inner names
    if len(inner) == 0:
        return []
    else:
        return [f"{outer}{names_sep}{name}" for name in inner]

def vec_paste0(*args):
    # Paste function that recycles arguments appropriately
    args = [np.repeat(arg, max(len(arg) for arg in args)) if len(arg) == 1 else arg for arg in args]
    return [''.join(str(items)) for items in zip(*args)]

def check_data_frame(x, arg='x'):
    # Check if x is a DataFrame
    if not isinstance(x, pd.DataFrame):
        raise Exception(f"`{arg}` must be a data frame, not {type(x).__name__}.")

def check_unique_names(x, arg='x'):
    # Check if all elements of x are named and unique
    if len(x) > 0 and not all(x.keys()):
        raise Exception(f"All elements of `{arg}` must be named.")
    if len(set(x.keys())) != len(x):
        raise Exception(f"The names of `{arg}` must be unique.")

def check_list_of_ptypes(x, names, arg='x'):
    # Check that x is a valid list of ptypes
    if x is None:
        return {name: None for name in names}
    elif isinstance(x, (pd.DataFrame, np.ndarray, list)) and len(x) == 0:
        return {name: x for name in names}
    elif isinstance(x, dict):
        check_unique_names(x, arg=arg)
        return {k: x[k] for k in x.keys() if k in names}
    else:
        raise Exception(f"`{arg}` must be `None`, an empty ptype, or a named dict of ptypes.")

def check_list_of_functions(x, names, arg='x'):
    # Check that x is a valid list of functions
    if x is None:
        x = {}
    elif callable(x):
        x = {name: x for name in names}
    elif isinstance(x, dict):
        check_unique_names(x, arg=arg)
    else:
        raise Exception(f"`{arg}` must be `None`, a function, or a named dict of functions.")
    x = {k: x[k] for k, v in x.items() if k in names}
    return x

def check_list_of_bool(x, names, arg='x'):
    # Check that x is a valid list of booleans
    if isinstance(x, bool):
        return {name: x for name in names}
    elif isinstance(x, dict):
        check_unique_names(x, arg=arg)
        x = {k: x[k] for k in x.keys() if k in names}
        return x
    else:
        raise Exception(f"`{arg}` must be a dict or a single `True` or `False`.")

def with_indexed_errors(expr, message, _error_call=None, _frame=None):
    # Execute expr and handle exceptions with custom messages
    try:
        return expr()
    except Exception as e:
        raise Exception(message(e)) from e

def int_max(x, default=float('-inf')):
    # Return the max of x, or default if x is empty
    if len(x) == 0:
        return default
    else:
        return max(x)