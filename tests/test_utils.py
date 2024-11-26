import pandas as pd

def test_that(description, test_fn):
    print(f"Testing: {description}")
    test_fn()
    print("...Test passed.")

def expect_equal(actual, expected):
    if isinstance(actual, pd.DataFrame):
        assert actual.equals(expected), f"Expected {expected}, got {actual}"
    else:
        assert actual == expected, f"Expected {expected}, got {actual}"

def tidyr_legacy(lst):
    output = []
    counts = {}
    empty_count = 1
    for i, name in enumerate(lst):
        if name == "":
            name = f"V{empty_count}"
            empty_count += 1
            output.append(name)
        elif name in counts:
            counts[name] += 1
            output.append(f"{name}{counts[name]-1}")
        else:
            counts[name] = 1
            output.append(name)
    return output

def test_tidyr_legacy(copies_old_approach):
    expect_equal(tidyr_legacy([]), [])
    expect_equal(tidyr_legacy(["x", "x", "y"]), ["x", "x1", "y"])
    expect_equal(tidyr_legacy(["", "", ""]), ["V1", "V2", "V3"])

def reconstruct_tibble(df_old, df_new):
    # This function should not repair names
    return df_new

def test_reconstruct_doesnt_repair_names():
    # This ensures that name repair elsewhere isn't overridden
    # Create DataFrame with duplicate column names
    df = pd.DataFrame([[1, 2]], columns=['x', 'x'])
    # Set attribute to simulate .name_repair = "minimal"
    df.attrs['_name_repair'] = 'minimal'
    # Now, test that reconstruct_tibble(df, df) is equal to df
    result = reconstruct_tibble(df, df)
    expect_equal(result, df)

# Run tests
test_that("tidyr_legacy copies old approach", test_tidyr_legacy)
test_that("reconstruct doesn't repair names", test_reconstruct_doesnt_repair_names)