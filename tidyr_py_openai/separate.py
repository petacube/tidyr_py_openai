import pandas as pd
import numpy as np
import re
import warnings

def separate(data, col, into, sep='[^\\w]+', remove=True, convert=False, extra='warn', fill='warn'):
    # Separate a column into multiple columns by splitting the values
    # data: pandas DataFrame
    # col: name of column to split
    # into: list of output column names
    # sep: separator (regex string or list of integers)
    # remove: whether to remove the original column
    # convert: whether to attempt to convert data types
    # extra: what to do with extra values ('warn', 'drop', or 'merge')
    # fill: what to do with missing values ('warn', 'left', or 'right')
    
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in data.")
    
    # Get the column values as string
    value = data[col].astype(str).values
    
    # Call str_separate
    new_cols = str_separate(value, into=into, sep=sep, convert=convert, extra=extra, fill=fill)
    
    # Add new_cols to data
    out = data.copy()
    for name in new_cols.columns:
        out[name] = new_cols[name]
    if remove:
        out = out.drop(columns=[col])
    return out


def str_separate(x, into, sep, convert=False, extra='warn', fill='warn'):
    # Helper function to split strings and return a DataFrame
    # x: array-like of strings
    # into: list of output column names
    # sep: separator (string regex or list of integers)
    # convert: whether to attempt to convert data types
    # extra: what to do with extra values ('warn', 'drop', or 'merge')
    # fill: what to do with missing values ('warn', 'left', or 'right')
    
    if isinstance(sep, (list, tuple, np.ndarray)) and all(isinstance(i, int) for i in sep):
        out = strsep(x, sep)
    elif isinstance(sep, str):
        n = len(into)
        out = str_split_fixed(x, sep, n=n, extra=extra, fill=fill)
    else:
        raise ValueError(f"sep must be a string or numeric vector, not {type(sep)}")
    
    # Create DataFrame from out
    out_df = pd.DataFrame(out)
    # Assign column names
    col_names = into.copy()
    if len(col_names) < out_df.shape[1]:
        col_names.extend([None]*(out_df.shape[1] - len(col_names)))
    out_df.columns = col_names
    
    # Remove columns where names are None or NA
    valid_cols = [name for name in col_names if name is not None and not pd.isna(name)]
    out_df = out_df.loc[:, valid_cols]
    
    if convert:
        out_df = out_df.apply(pd.to_numeric, errors='ignore')
    
    return out_df


def strsep(x, sep):
    # Split strings at specified positions
    # x: array of strings
    # sep: list of integer positions
    # Returns: list of lists (splits for each string)
    result = []
    for s in x:
        if pd.isna(s) or s == 'nan':
            substrings = [np.nan]*(len(sep)+1)
        else:
            s_len = len(s)
            positions = []
            for i in sep:
                if i >= 0:
                    pos = i
                else:
                    pos = max(0, s_len + i)
                positions.append(pos)
            positions = [0] + positions + [s_len]
            positions = sorted(set(positions))
            substrings = [s[positions[i]:positions[i+1]] for i in range(len(positions)-1)]
        result.append(substrings)
    return result


def str_split_fixed(value, sep, n, extra='warn', fill='warn'):
    # Split strings using regex pattern, with fixed number of splits
    # value: array of strings
    # sep: regex pattern
    # n: expected number of pieces
    # extra: 'warn', 'drop', 'merge'
    # fill: 'warn', 'left', 'right'
    
    if extra == 'error':
        warnings.warn('`extra = "error"` is deprecated. Please use `extra = "warn"` instead')
        extra = 'warn'
    if extra not in ['warn', 'drop', 'merge']:
        raise ValueError(f"Invalid value for extra: {extra}")
    if fill not in ['warn', 'left', 'right']:
        raise ValueError(f"Invalid value for fill: {fill}")
    
    n_max = n if extra == 'merge' else -1
    pieces = str_split_n(value, sep, n_max=n_max)
    simplified = simplify_pieces(pieces, n, extra=extra, fill=fill)
    return simplified


def str_split_n(x, pattern, n_max=-1):
    # Split strings using regex, with max number of splits
    # x: array of strings
    # pattern: regex pattern
    # n_max: maximum number of splits (-1 means no limit)
    result = []
    for s in x:
        if pd.isna(s) or s == 'nan':
            result.append([np.nan])
        else:
            splits = re.split(pattern, s, maxsplit=n_max)
            result.append(splits)
    return result


def simplify_pieces(pieces, n, extra='warn', fill='warn'):
    # Simplify list of splits to have length n, handling extra and missing pieces
    # pieces: list of lists (splits for each string)
    # n: expected number of pieces
    # extra: 'warn', 'drop', 'merge'
    # fill: 'warn', 'left', 'right'
    too_big = []
    too_small = []
    simplified = []
    for idx, splits in enumerate(pieces):
        length = len(splits)
        if length > n:
            # Handle extra pieces
            if extra == 'merge':
                # Merge extra pieces into the last piece
                last_piece = ''.join(splits[n-1:])
                splits = splits[:n-1] + [last_piece]
            elif extra == 'drop':
                splits = splits[:n]
            elif extra == 'warn':
                splits = splits[:n]
                too_big.append(idx)
        elif length < n:
            # Handle missing pieces
            missing = n - length
            if fill == 'left':
                splits = [np.nan]*missing + splits
            elif fill == 'right':
                splits = splits + [np.nan]*missing
            elif fill == 'warn':
                splits = splits + [np.nan]*missing
                too_small.append(idx)
        simplified.append(splits)
    if (extra == 'warn') and too_big:
        idx_str = list_indices(too_big)
        warnings.warn(f"Expected {n} pieces. Additional pieces discarded in {len(too_big)} rows [{idx_str}].")
    if (fill == 'warn') and too_small:
        idx_str = list_indices(too_small)
        warnings.warn(f"Expected {n} pieces. Missing pieces filled with `NA` in {len(too_small)} rows [{idx_str}].")
        
    return simplified


def list_indices(x, max_items=20):
    # Convert list of indices to string representation
    # x: list of zero-based indices
    x = [i+1 for i in x]  # Convert to one-based indices
    if len(x) > max_items:
        x = x[:max_items] + ['...']
    return ', '.join(map(str, x))