import pandas as pd
import numpy as np

def uncount(data, weights, *args, remove=True, id=None):
    # Performs the opposite operation to pandas.DataFrame.value_counts(), duplicating rows
    # according to a weighting variable (or expression).
    #
    # data: A pandas DataFrame.
    # weights: A vector of weights. Evaluated in the context of data; supports expressions.
    # *args: Additional arguments (not used).
    # remove: If True, and weights is the name of a column in data, then this column is removed.
    # id: Supply a string to create a new variable which gives a unique identifier for each created row.

    # Check that data is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")

    # Validate remove parameter
    if not isinstance(remove, bool):
        raise ValueError("remove must be a boolean")

    # Validate id parameter
    if id is not None and not isinstance(id, str):
        raise ValueError("id must be a string or None")

    # Evaluate weights in the context of data
    # weights may be a callable, an expression string, a column name, a Series, or a constant

    is_symbol = False

    if callable(weights):
        # weights is a function, apply it to data
        w = data.apply(weights, axis=1)
    elif isinstance(weights, str):
        # Try to evaluate weights as an expression in data
        try:
            w = data.eval(weights)
        except Exception:
            # If it fails, assume weights is column name
            if weights in data.columns:
                w = data[weights]
                is_symbol = True
            else:
                raise ValueError(f"Cannot evaluate weights '{weights}' in data")
    elif isinstance(weights, (int, float)):
        # weights is a constant
        w = pd.Series([weights] * len(data), index=data.index)
    elif isinstance(weights, (pd.Series, np.ndarray, list)):
        # weights is a Series or list
        if len(weights) != len(data):
            raise ValueError("Length of weights must match number of rows in data")
        w = pd.Series(weights, index=data.index)
    else:
        raise ValueError("Unsupported type for weights")

    # Convert weights to numeric, fill NaN with 0, and cast to integer
    w_int = pd.to_numeric(w, errors='coerce').fillna(0).astype(int)
    w_int = w_int.clip(lower=0)  # Ensure weights are non-negative

    # Replicate rows according to weights
    out = data.loc[data.index.repeat(w_int)].reset_index(drop=True)

    # Remove weights column if required
    if remove and is_symbol:
        if weights in out.columns:
            out = out.drop(columns=weights)

    # Add id column if specified
    if id is not None:
        # Generate sequence numbers for each row replication
        sequence_numbers = [i for count in w_int for i in range(1, count+1)]
        out[id] = sequence_numbers

    return out