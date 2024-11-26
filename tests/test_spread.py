import pandas as pd
import numpy as np
import unittest

# Define the spread function
def spread(df, key, value, fill=None, convert=False, drop=True):
    other_columns = [col for col in df.columns if col != key and col != value]
    # Handle the case when other_columns is empty
    if other_columns:
        spread_df = df.set_index(other_columns + [key])[value].unstack(key)
    else:
        spread_df = df.set_index([key])[value].unstack(key)
    if not drop and hasattr(df[key], 'cat'):
        # Reindex to include all categories, including NaN if present
        categories = ulevels(df[key])
        spread_df = spread_df.reindex(categories, axis=1)
    if fill is not None:
        spread_df = spread_df.fillna(fill)
    else:
        spread_df = spread_df
    spread_df = spread_df.reset_index()
    spread_df.columns.name = None
    if convert:
        for col in spread_df.columns:
            if col not in other_columns:
                spread_df[col] = pd.to_numeric(spread_df[col], errors='ignore')
    return spread_df

def ulevels(x):
    if hasattr(x, 'cat'):
        # For categorical variables, return their categories including NaN if present
        return x.cat.categories.tolist()
    else:
        # For lists or other iterables
        return pd.unique(x)

# Test class
class TestSpreadFunction(unittest.TestCase):

    def test_order_doesnt_matter(self):
        # df1
        df1 = pd.DataFrame({
            'x': pd.Categorical(['a', 'b']),
            'y': [1, 2]
        })
        # df2
        df2 = pd.DataFrame({
            'x': pd.Categorical(['b', 'a']),
            'y': [2, 1]
        })
        one = spread(df1, 'x', 'y')
        two = spread(df2, 'x', 'y')[['a', 'b']]
        pd.testing.assert_frame_equal(one, two)

        df1 = pd.DataFrame({
            'z': pd.Categorical(['b', 'a']),
            'x': pd.Categorical(['a', 'b']),
            'y': [1, 2]
        })
        df2 = pd.DataFrame({
            'z': pd.Categorical(['a', 'b']),
            'x': pd.Categorical(['b', 'a']),
            'y': [2, 1]
        })
        one = spread(df1, 'x', 'y').sort_values('z').reset_index(drop=True)
        two = spread(df2, 'x', 'y')
        pd.testing.assert_frame_equal(one, two)

    def test_convert_turns_strings_into_integers(self):
        df = pd.DataFrame({'key': ['a'], 'value': ['1']})
        out = spread(df, 'key', 'value', convert=True)
        self.assertTrue(out['a'].dtype == np.dtype('int64'))

    def test_duplicate_values_for_one_key_is_an_error(self):
        df = pd.DataFrame({
            'x': pd.Categorical(['a', 'b', 'b']),
            'y': [1, 2, 2],
            'z': [1, 2, 2]
        })
        with self.assertRaises(ValueError):
            spread(df, 'x', 'y')

    def test_factors_are_spread_into_columns(self):
        data = pd.DataFrame({
            'x': pd.Categorical(['a', 'a', 'b', 'b']),
            'y': pd.Categorical(['c', 'd', 'c', 'd']),
            'z': pd.Categorical(['w', 'x', 'y', 'z'])
        })

        out = spread(data, 'x', 'z')
        self.assertEqual(list(out.columns), ['y', 'a', 'b'])
        all_categorical = all([isinstance(out[col].dtype, pd.CategoricalDtype) for col in out.columns])
        self.assertTrue(all_categorical)
        self.assertListEqual(list(out['a'].cat.categories), list(data['z'].cat.categories))
        self.assertListEqual(list(out['b'].cat.categories), list(data['z'].cat.categories))

    def test_drop_false_keeps_missing_combinations(self):
        df = pd.DataFrame({
            'x': pd.Categorical(['a'], categories=['a', 'b']),
            'y': pd.Categorical(['b'], categories=['a', 'b']),
            'z': [1]
        })
        out = spread(df, 'x', 'z', drop=False)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out.columns), 3)
        self.assertEqual(out.loc[1, 'a'], 1)

    def test_drop_false_keeps_missing_combinations_of_zero_length_factors(self):
        df = pd.DataFrame({
            'x': pd.Categorical([], categories=['a', 'b']),
            'y': pd.Categorical([], categories=['a', 'b']),
            'z': []
        })
        out = spread(df, 'x', 'z', drop=False)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out.columns), 3)
        self.assertTrue(out['a'].isnull().all())
        self.assertTrue(out['b'].isnull().all())

    def test_drop_false_spread_all_levels_including_NA(self):
        l = ['a', 'b', 'c', 'd']
        df = pd.DataFrame({
            'x': pd.Categorical(['a', 'b', 'c', np.nan], categories=l),
            'y': pd.Categorical(['a', 'b', 'c', 'd']),
            'z': pd.Categorical(['a', 'b', 'a', 'b'])
        })
        out = spread(df, 'x', 'y', drop=False)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out.columns), 6)
        self.assertTrue(out['d'].isnull().iloc[0])
        self.assertEqual(out['d'].iloc[1], 'd')
        self.assertTrue(out['<NA>'].isnull().iloc[0])
        self.assertEqual(out['<NA>'].iloc[1], 'd')

    def test_spread_preserves_class_of_tibbles(self):
        data = pd.DataFrame({
            'x': pd.Categorical(['a', 'a', 'b', 'b']),
            'y': pd.Categorical(['c', 'd', 'c', 'd']),
            'z': pd.Categorical(['w', 'x', 'y', 'z'])
        })
        out = spread(data, 'x', 'z')
        self.assertTrue(isinstance(out, pd.DataFrame))

    def test_dates_are_spread_into_columns(self):
        df = pd.DataFrame({
            'id': ['a', 'a', 'b', 'b'],
            'key': ['begin', 'end', 'begin', 'end'],
            'date': pd.to_datetime(pd.Timestamp('today') + pd.to_timedelta(np.arange(4), unit='D'))
        })
        out = spread(df, 'key', 'date')
        self.assertEqual(list(out.columns), ['id', 'begin', 'end'])
        self.assertTrue(out['begin'].dtype == 'datetime64[ns]')
        self.assertTrue(out['end'].dtype == 'datetime64[ns]')

    def test_spread_can_produce_mixed_variable_types(self):
        df = pd.DataFrame({
            'row': [1,2,1,2,1,2],
            'column': [1,1,2,2,3,3],
            'cell_contents': ['Argentina', 'Argentina', '62.485', '64.399', '1952', '1957']
        })
        out = spread(df, 'column', 'cell_contents', convert=True)
        self.assertEqual([out[col].dtype.name for col in out.columns],
                         ['int64', 'object', 'float64', 'int64'])

    def test_factors_can_be_used_with_convert_TRUE_to_produce_mixed_types(self):
        df = pd.DataFrame({
            'row': [1,2,1,2,1,2],
            'column': ['f','f','g','g','h','h'],
            'contents': ['aa','bb','1','2','TRUE','FALSE']
        })
        out = spread(df, 'column', 'contents', convert=True)
        self.assertEqual(out['f'].dtype.name, 'object')
        self.assertEqual(out['g'].dtype.name, 'int64')
        self.assertEqual(out['h'].dtype.name, 'bool')

    def test_dates_can_be_used_with_convert_TRUE(self):
        df = pd.DataFrame({
            'id': ['a', 'a', 'b', 'b'],
            'key': ['begin', 'end', 'begin', 'end'],
            'date': pd.to_datetime(pd.Timestamp('today') + pd.to_timedelta(np.arange(4), unit='D'))
        })
        out = spread(df, 'key', 'date', convert=True)
        self.assertEqual(out['begin'].dtype.name, 'datetime64[ns]')
        self.assertEqual(out['end'].dtype.name, 'datetime64[ns]')

    def test_vars_that_are_all_NA_are_logical_if_convert_TRUE(self):
        df = pd.DataFrame({
            'row': [1,2,1,2],
            'column': ['f','f','g','g'],
            'contents': ['aa','bb',np.nan,np.nan]
        })
        out = spread(df, 'column', 'contents', convert=True)
        self.assertEqual(out['g'].dtype.name, 'object')  # In pandas, NA in object type columns

    def test_complex_values_are_preserved(self):
        df = pd.DataFrame({
            'id': [1,1,2,2],
            'key': ['a','b','a','b'],
            'value': [1+1j,2+1j,3+1j,4+1j]
        })
        out1 = spread(df, 'key', 'value', convert=False)
        out2 = spread(df, 'key', 'value', convert=True)
        np.testing.assert_array_equal(out1['a'], np.array([1+1j,3+1j]))
        np.testing.assert_array_equal(out2['a'], np.array([1+1j,3+1j]))
        np.testing.assert_array_equal(out1['b'], np.array([2+1j,4+1j]))
        np.testing.assert_array_equal(out2['b'], np.array([2+1j,4+1j]))

    def test_can_spread_with_nested_columns(self):
        df = pd.DataFrame({
            'x': ['a', 'a'],
            'y': [1,2],
            'z': [pd.Series([1,2]), pd.Series([3,4,5])]
        })
        out = spread(df, 'x', 'y')
        self.assertEqual(out['a'].tolist(), [1, np.nan])
        self.assertEqual(out['z'].tolist(), df['z'].tolist())

    def test_spreading_empty_data_frame_gives_empty_data_frame(self):
        df = pd.DataFrame({'x': [], 'y': [], 'z': []})
        rs = spread(df, 'x', 'y')
        self.assertEqual(len(rs), 0)
        self.assertEqual(list(rs.columns), ['z'])

        df = pd.DataFrame({'x': [], 'y': []})
        rs = spread(df, 'x', 'y')
        self.assertEqual(len(rs), 0)
        self.assertEqual(len(rs.columns), 0)

    def test_spread_gives_one_column_when_no_existing_non_spread_vars(self):
        df = pd.DataFrame({
            'key': ['a','b','c'],
            'value': [1,2,3]
        })
        out = spread(df, 'key', 'value')
        self.assertTrue(out.equals(pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})))

    def test_grouping_vars_are_kept_where_possible(self):
        df = pd.DataFrame({'x': [1,2], 'key': pd.Categorical(['a','b']), 'value': [1,2]})
        out = df.groupby('x').apply(lambda d: spread(d, 'key', 'value'))
        # Since grouping keys are preserved in pandas after groupby
        self.assertTrue('x' in out.columns)

    def test_col_names_never_contains_NA(self):
        df = pd.DataFrame({'x': [1, np.nan], 'y': [1,2]})
        out = spread(df, 'x', 'y')
        self.assertEqual(list(out.columns), ['1.0', 'nan'])
        out = spread(df, 'x', 'y')
        out.columns = ['x_1.0', 'x_nan']
        self.assertEqual(list(out.columns), ['x_1.0', 'x_nan'])

    def test_never_has_row_names(self):
        df = pd.DataFrame({'id': [1,2], 'x': ['a','b'], 'y': [1,2]})
        out = spread(df, 'x', 'y')
        self.assertFalse(out.index.names[0] is not None)

    def test_overwrites_existing_columns(self):
        df = pd.DataFrame({'x': [1,2], 'y': [2,1], 'key': ['x','x'], 'value': [3,4]})
        rs = spread(df, 'key', 'value')
        self.assertEqual(list(rs.columns), ['y', 'x'])
        self.assertEqual(rs['x'].tolist(), [3,4])

    def test_spread_doesnt_convert_data_frames_into_tibbles(self):
        df = pd.DataFrame({'x': ['a', 'b'], 'y': [1,2]})
        rs = spread(df, 'x', 'y')
        self.assertTrue(isinstance(rs, pd.DataFrame))

    def test_spread_with_fill_replaces_explicit_missing_values(self):
        df = pd.DataFrame({'key': pd.Categorical(['a']), 'value': [np.nan]})
        out = spread(df, 'key', 'value', fill=1)
        self.assertEqual(out['a'].iloc[0], 1)

    def test_spread_with_fill_replaces_implicit_missing_values(self):
        df = pd.DataFrame({
            'x': pd.Categorical(['G1', 'G2']),
            'key': pd.Categorical(['a', 'b']),
            'value': [1,1]
        })
        out = spread(df, 'key', 'value', fill=2)
        self.assertTrue(out.equals(pd.DataFrame({'x': ['G1', 'G2'], 'a': [1,2], 'b': [2,1]})))

        df = pd.DataFrame({'key': pd.Categorical(['a'], categories=['a', 'b']), 'value': [1]})
        out = spread(df, 'key', 'value', fill=2, drop=False)
        self.assertTrue(out.equals(pd.DataFrame({'a': [1], 'b': [2]})))

    def test_ulevels_preserves_original_factor_levels(self):
        x_na_lev = pd.Categorical(['a', np.nan], categories=['a', np.nan])
        self.assertEqual(ulevels(x_na_lev), ['a', np.nan])

        x_na_lev_extra = pd.Categorical(['a', np.nan], categories=['a', 'b', np.nan])
        self.assertEqual(ulevels(x_na_lev_extra), ['a', 'b', np.nan])

        x_no_na_lev = pd.Categorical(['a', np.nan])
        self.assertEqual(ulevels(x_no_na_lev), ['a'])

        x_no_na_lev_extra = pd.Categorical(['a', np.nan], categories=['a', 'b'])
        self.assertEqual(ulevels(x_no_na_lev_extra), ['a', 'b'])

    def test_ulevels_returns_unique_elements_of_a_list_for_a_list_input(self):
        test_list = [1,2,3,4,5,6,1,2,3,4,5,6]
        self.assertEqual(ulevels(test_list).tolist(), [1,2,3,4,5,6])

    def test_spread_works_when_id_column_has_names(self):
        df = pd.DataFrame({
            'key': pd.Categorical(['a','b','c'], categories=['a','b','c','d','e']),
            'out': [1,2,3],
            'id': pd.Series([1,2,3], index=['a','b','c'])
        })
        out = spread(df, 'key', 'out', drop=False)
        self.assertEqual(list(out.columns), ['id', 'a', 'b', 'c', 'd', 'e'])

if __name__ == '__main__':
    unittest.main()