import pytest
import numpy as np
import torch
import os
import shutil

from mi_toolbox.utils.data_types import DataDict

class TestDataDict:

    @pytest.fixture
    def basic_dd(self):
        """Returns a simple DataDict populated with python lists."""
        data = [
            {'id': 0, 'val': 'a', 'score': 1.1},
            {'id': 1, 'val': 'b', 'score': 2.2},
            {'id': 2, 'val': 'c', 'score': 3.3}
        ]
        return DataDict.from_list(data)

    @pytest.fixture
    def mixed_dd(self):
        """Returns a DataDict with Lists, NumPy arrays, and Torche Tensors."""
        dd = DataDict()
        dd.attach("list_col", [1, 2, 3])
        dd.attach("np_col", np.array([10, 20, 30]))
        dd.attach("pt_col", torch.tensor([100, 200, 300]))
        return dd

    # --- Initialization Tests ---

    def test_init_empty(self):
        dd = DataDict()
        assert len(dd) == 0
        assert dd.data == {}

    def test_init_columns(self):
        dd = DataDict(columns=['a', 'b'])
        assert dd.data['a'] == []
        assert dd.data['b'] == []
        assert len(dd) == 0

    def test_from_list(self):
        data = [{'a': 1}, {'a': 2}]
        dd = DataDict.from_list(data)
        assert len(dd) == 2
        assert dd['a'] == [1, 2]

    def test_from_dict(self):
        data = {'col1': [1, 2], 'col2': [3, 4]}
        dd = DataDict.from_dict(data)
        assert len(dd) == 2
        assert dd['col1'] == [1, 2]

    # --- CRUD & Ingestion Tests ---

    def test_append(self):
        dd = DataDict(columns=['a'])
        dd.append({'a': 1})
        assert len(dd) == 1
        assert dd['a'] == [1]

        # Append with missing key (should be None)
        dd.attach('b', [10], force=True) # First fix structure
        dd.append({'a': 2}) # Missing 'b'
        assert dd['b'][1] is None

    def test_extend(self):
        dd = DataDict(columns=['a'])
        dd.extend({'a': [1, 2, 3]})
        assert len(dd) == 3
        assert dd['a'] == [1, 2, 3]

    def test_attach_mismatch(self):
        dd = DataDict.from_list([{'a': 1}, {'a': 2}])
        with pytest.raises(ValueError):
            dd.attach("new_col", [1, 2, 3]) # Length 3 vs 2

    # --- Indexing (__getitem__) Tests ---

    def test_getitem_column(self, basic_dd):
        col = basic_dd['val']
        assert col == ['a', 'b', 'c']

    def test_getitem_row_int(self, basic_dd):
        row = basic_dd[1]
        assert row['id'] == 1
        assert row['val'] == 'b'
        
        # Test negative index
        row_neg = basic_dd[-1]
        assert row_neg['id'] == 2

    def test_getitem_slice(self, basic_dd):
        subset = basic_dd[0:2]
        assert isinstance(subset, DataDict)
        assert len(subset) == 2
        assert subset['id'] == [0, 1]

    def test_getitem_list_rows(self, basic_dd):
        # Select rows 0 and 2
        subset = basic_dd[[0, 2]]
        assert len(subset) == 2
        assert subset['val'] == ['a', 'c']

    def test_getitem_list_cols(self, basic_dd):
        # Select subset of columns
        subset = basic_dd[['id', 'score']]
        assert isinstance(subset, DataDict)
        assert list(subset.keys()) == ['id', 'score']
        assert len(subset) == 3

    def test_getitem_tensor_indices(self, mixed_dd):
        # Select using torch tensor indices
        indices = torch.tensor([0, 2])
        subset = mixed_dd[indices]
        assert len(subset) == 2
        assert subset['list_col'] == [1, 3]
        assert torch.equal(subset['pt_col'], torch.tensor([100, 300]))

    # --- Modification (__setitem__) Tests ---

    def test_setitem_column_replace(self, basic_dd):
        new_col = [10, 11, 12]
        basic_dd['id'] = new_col
        assert basic_dd['id'] == new_col

    def test_setitem_row_update(self, basic_dd):
        basic_dd[0] = {'id': 999, 'val': 'z', 'score': 0.0}
        assert basic_dd['id'][0] == 999
        assert basic_dd['val'][0] == 'z'

    def test_setitem_slice(self, basic_dd):
        # Update first two rows
        basic_dd[0:2] = {'val': ['x', 'y']}
        assert basic_dd['val'] == ['x', 'y', 'c']

    def test_setitem_multi_row(self, basic_dd):
        # Update rows 0 and 2
        updates = [
            {'id': 100, 'val': 'updated_a', 'score': 0},
            {'id': 200, 'val': 'updated_c', 'score': 0}
        ]
        basic_dd[[0, 2]] = updates
        assert basic_dd['id'] == [100, 1, 200]

    # --- Sorting Tests ---

    def test_sort_simple(self):
        dd = DataDict.from_list([{'a': 3}, {'a': 1}, {'a': 2}])
        dd.sort('a')
        assert dd['a'] == [1, 2, 3]

    def test_sort_descending(self):
        dd = DataDict.from_list([{'a': 1}, {'a': 3}, {'a': 2}])
        dd.sort('a', descending=True)
        assert dd['a'] == [3, 2, 1]

    def test_sort_mixed_types(self, mixed_dd):
        # Force unsorted state
        mixed_dd['list_col'] = [3, 1, 2]
        mixed_dd['np_col'] = np.array([30, 10, 20])
        mixed_dd['pt_col'] = torch.tensor([300, 100, 200])

        mixed_dd.sort('list_col')
        
        assert mixed_dd['list_col'] == [1, 2, 3]
        # Check if numpy and torch columns followed the sort
        assert np.array_equal(mixed_dd['np_col'], np.array([10, 20, 30]))
        assert torch.equal(mixed_dd['pt_col'], torch.tensor([100, 200, 300]))

    def test_sort_multi_key(self):
        # Sort by 'group' then 'val'
        data = [
            {'group': 2, 'val': 10},
            {'group': 1, 'val': 50},
            {'group': 1, 'val': 5}
        ]
        dd = DataDict.from_list(data)
        dd.sort(['group', 'val'])
        
        assert dd['group'] == [1, 1, 2]
        assert dd['val'] == [5, 50, 10]

    # --- Map / Concurrency Tests ---

    def test_map_inplace(self):
        dd = DataDict.from_list([{'val': 1}, {'val': 2}])
        
        def double_val(idx, row):
            return {'val': row['val'] * 2}
        
        dd.map(double_val, inplace=True)
        assert dd['val'] == [2, 4]

    def test_map_return_new(self):
        dd = DataDict.from_list([{'val': 1}, {'val': 2}, {'val': 3}])
        
        # Filter: only keep even numbers
        def filter_even(idx, row):
            if row['val'] % 2 == 0:
                return row
            return None # Drop odd rows
        
        new_dd = dd.map(filter_even, inplace=False)
        assert len(new_dd) == 1
        assert new_dd['val'] == [2]
        
        # Original should be untouched
        assert len(dd) == 3

    # --- Save / Load Tests ---

    def test_save_load(self, tmp_path, mixed_dd):
        save_dir = tmp_path / "dd_storage"
        
        # Save
        mixed_dd.save(str(save_dir))
        
        # Load back
        keys = ['list_col', 'np_col', 'pt_col']
        loaded_dd = DataDict.load(str(save_dir), keys)
        
        assert len(loaded_dd) == 3
        assert loaded_dd['list_col'] == [1, 2, 3]
        # Note: Load logic converts numpy -> torch tensor during save? 
        # Looking at _save implementation:
        # np.ndarray -> torch.save(torch.from_numpy)
        # So loaded data will be torch tensors for both np and pt
        
        assert isinstance(loaded_dd['np_col'], torch.Tensor)
        assert torch.equal(loaded_dd['np_col'], torch.tensor([10, 20, 30]))
        assert torch.equal(loaded_dd['pt_col'], torch.tensor([100, 200, 300]))

    # --- Error Handling Tests ---

    def test_length_mismatch_error(self):
        dd = DataDict.from_list([{'a': 1}])
        with pytest.raises(ValueError):
            dd['new_col'] = [1, 2] # Mismatch length

    def test_invalid_key_type(self, basic_dd):
        with pytest.raises(TypeError):
            _ = basic_dd[1.5] # Float index not supported for single row

    def test_row_missing_keys_error(self, basic_dd):
        with pytest.raises(KeyError):
            # Attempting to add a row with a key that doesn't exist in columns
            basic_dd[0] = {'id': 1, 'val': 'a', 'score': 1.1, 'EXTRA': 999}