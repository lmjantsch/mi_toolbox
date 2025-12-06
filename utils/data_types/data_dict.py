import os
import json
from typing import Union, List, Dict, Any, Optional, Iterator
from collections.abc import KeysView, ItemsView, ValuesView
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

ColumnType = Union[List, np.ndarray, torch.Tensor]
RowType = Dict[str, Any]

class DataDict:

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        """
        Initialize the DataDict.
        
        Args:
            columns: A list of column names to initialize empty columns. 
                     Does NOT accept actual data.
        """
        self.data: Dict[str, ColumnType] = {}
        self.length: int = 0

        if columns:
            if not isinstance(columns, list) or not all(isinstance(x, str) for x in columns):
                raise TypeError("columns must be a list of strings")
            self.data = {col: [] for col in columns}

    def __getitem__(self, key: Union[int, str, slice, List[str], List[int]]) -> Union['DataDict', ColumnType, RowType]:
        # String (Column Selection)
        if isinstance(key, str):
            if key not in self.data:
                raise KeyError(f"Column '{key}' not found.")
            return self.data[key]
        
        # Int (Row Selection)
        if isinstance(key, int):
            if key < 0:
                key += self.length
            if key < 0 or key >= self.length:
                raise IndexError("Row index out of range")
            return {k: self.data[k][key] for k in self.data}
    
        # Slice (Row Selection) -> returns DataDict
        if isinstance(key, slice):
            return self.from_dict({k:self.data[k][key] for k in self.data})
        
        # List (DataDict Subset)
        if isinstance(key, (list, np.ndarray, torch.Tensor)):
            # Convert to list if it's tensor/array for easier iteration checks
            if isinstance(key, (np.ndarray, torch.Tensor)):
                key = key.tolist()
            
            if not key:
                return type(self)()

            # Case A: List of Strings (Column Select)
            if isinstance(key[0], str):
                return self.from_dict({k: self.data[k] for k in key})
            
            # Case B: List of Ints (Row Select)
            if isinstance(key[0], (int, float)): # float handled for numpy int types
                 new_data = {}
                 for k, v in self.data.items():
                     if isinstance(v, (np.ndarray, torch.Tensor)):
                         new_data[k] = v[key]
                     else:
                         new_data[k] = [v[i] for i in key]
                 return self.from_dict(new_data)
            
        raise TypeError(f"Unsupported key type: {type(key)}")
    
    def __setitem__(self, key: Union[int, str, slice], value: any) -> None:
        # Edit/Add Column (str)
        if isinstance(key, str):
            if not isinstance(value, (list, np.ndarray, torch.Tensor)):
                raise TypeError("Expected List, Array, or Tensor for column editing")
            
            if self.length == 0 and not self.data:
                self.length = len(value)
            
            if len(value) != self.length:
                raise ValueError(f"Length mismatch: New column '{key}' has {len(value)}, expected {self.length}.")
            
            self.data[key] = value
            return
        
        # Edit Row (int)
        if isinstance(key, int):
            if not isinstance(value, dict):
                raise TypeError('Expected dictionary for row editing')
            
            if key < 0: 
                key += self.length
            if key >= self.length:
                raise IndexError("Row index out of range")

            if not set(value).issubset(self.data.keys()):
                raise KeyError(f'Unexpected Keys: {list(set(value) - set(self.data.keys()))}')
            
            for k, v in value.items():
                self.data[k][key] = v
            return
        
        # Edit Slice
        if isinstance(key, slice):
            if not isinstance(value, dict):
                raise TypeError('Expected dictionary of columns for slice editing')
            
            start, stop, step = key.indices(self.length)
            slice_range = range(start, stop, step)
            slice_len = len(slice_range)

            for col_k, col_v in value.items():
                if len(col_v) != slice_len:
                     raise ValueError(f"Length mismatch for column {col_k}: got {len(col_v)}, expected {slice_len}")
                self.data[col_k][key] = col_v
            return

        # Multi-Row/Column Edit (List)
        if isinstance(key, (list, np.ndarray, torch.Tensor)):
            if isinstance(key, (np.ndarray, torch.Tensor)):
                key = key.tolist()
            
            if not key: return

            # Multi-Column Edit
            if isinstance(key[0], str):
                if not isinstance(value, (list, tuple)) or len(value) != len(key):
                    raise ValueError("Value must be list of columns with matching length")
                for col_name, col_data in zip(key, value):
                    self[col_name] = col_data
                return

            # Multi-Row Edit
            if isinstance(key[0], int):
                if not isinstance(value, list) or len(value) != len(key):
                    raise ValueError("Value must be list of dicts with matching length")
                for row_idx, row_data in zip(key, value):
                    self[row_idx] = row_data
                return

        raise TypeError(f"Unsupported key type: {type(key)}")

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator:
        return iter(self.data.keys())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(length={self.length}, columns={list(self.data.keys())})"
    
    def keys(self) -> KeysView:
        return self.data.keys()
    
    def items(self) -> ItemsView:
        return self.data.items()

    def values(self) -> ValuesView:
        return self.data.values()
    
    def append(self, data: Dict[str, Any]) -> None:
        """Appends a single row."""
        if not isinstance(data, dict):
            raise TypeError(f"Dictionary expected but found {type(data)}.")
        
        if self.length == 0:
            self._add_data({k: [v] for k, v in data.items()})
            return

        if not set(data).issubset(self.data.keys()):
             raise KeyError(f'Unexpected Keys: {list(set(data) - set(self.data.keys()))}')

        for k in self.data:
            val = data.get(k, None)
            col = self.data[k]
            col.append(val)
        
        self.length += 1
    
    def extend(self, data: Dict[str, List]) -> None:
        """
        Extends with multiple rows (column-wise dict).
        Much faster than calling append() in a loop.
        """
        self._add_data(data)

    def attach(self, key: str, data: ColumnType, force: bool = False) -> None:
        """Attaches a new column to the dataset."""
        if not force and key in self.data:
            raise ValueError(f"The key '{key}' already exists.")
        
        # If this is the first attachment, it defines the object length
        if self.length == 0 and not any(len(v) > 0 for v in self.data.values()):
            self.length = len(data)
        
        if len(data) != self.length:
            raise ValueError(f"Length mismatch: New column has {len(data)}, expected {self.length}.")
        
        self.data[key] = data
    
    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> 'DataDict':
        """Creates DataDict from a list of dictionaries (rows)."""
        if not data:
            return cls()
        
        keys = list(data[0].keys())
        obj = cls(columns=keys)
        
        column_data = {k: [] for k in keys}
        for i, sample in enumerate(data):
            for k in keys:
                if k not in sample:
                     raise KeyError(f"Missing key '{k}' in sample {i}")
                column_data[k].append(sample[k])
        
        obj.data = column_data
        obj.length = len(data)
        return obj
    
    @classmethod
    def from_dict(cls, data: Dict[str, ColumnType]) -> 'DataDict':
        """Creates DataDict from a dictionary of lists (columns)."""
        obj = cls()
        if data:
            obj._add_data(data)
        return obj
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Converts the DataDict to a list of dictionaries (rows)."""
        return [{k: self.data[k][i] for k in self.data} for i in range(self.length)]

    def to_dict(self) -> Dict[str, ColumnType]:
        """Returns the internal dictionary of columns."""
        return self.data.copy()
    
    def _add_data(self, data: Dict[str, ColumnType]) -> None:
        if not data:
            return

        first_len = len(next(iter(data.values())))
        for k, v in data.items():
            if len(v) != first_len:
                raise ValueError(f"Column '{k}' length ({len(v)}) does not match others ({first_len})")

        if self.length == 0:
            self.data = dict(data)
            self.length = first_len
            return

        if not set(data.keys()).issubset(self.data.keys()):
             raise KeyError(f"New data contains unknown keys: {set(data.keys()) - set(self.data.keys())}")
        
        if first_len == 0:
            return

        for k in self.data:
            new_vals = data.get(k, None)
            if new_vals is None:
                 raise ValueError(f"Missing data for column '{k}'")

            current = self.data[k]
            
            if isinstance(current, list):
                if isinstance(new_vals, list):
                    current.extend(new_vals)
                else:
                    current.extend(list(new_vals))
            elif isinstance(current, np.ndarray):
                self.data[k] = np.concatenate((current, new_vals), axis=0)
            elif isinstance(current, torch.Tensor):
                if not isinstance(new_vals, torch.Tensor):
                    new_vals = torch.tensor(new_vals, device=current.device, dtype=current.dtype)
                self.data[k] = torch.cat((current, new_vals), dim=0)

        self.length += first_len
    
    def sort(self, by: Union[List[str], str], descending: bool = False) -> None:
        if self.length == 0:
            return

        if isinstance(by, str):
            by = [by]
        
        final_indices = np.arange(self.length)

        # iterate from least significant to most significant
        for key in reversed(by):
            col = self.data[key]
            
            # Tensors (GPU/CPU)
            if isinstance(col, torch.Tensor):
                current_indices_tensor = torch.from_numpy(final_indices).to(col.device)
                current_values = col[current_indices_tensor]
                
                local_indices = torch.argsort(current_values, descending=descending, stable=True)
                final_indices = final_indices[local_indices.cpu().numpy()]

            # NumPy Arrays
            elif isinstance(col, np.ndarray):
                current_values = col[final_indices]
                
                # fall back on python argsort for decending
                if descending:
                    current_values_list = current_values.tolist()
                    local_indices = sorted(range(len(current_values_list)), key=lambda i: current_values_list[i], reverse=True)
                    local_indices = np.array(local_indices)
                else:
                    local_indices = np.argsort(current_values, kind='stable')
                
                final_indices = final_indices[local_indices]

            # Lists
            else:
                current_values = [col[i] for i in final_indices]
                local_indices = sorted(range(len(current_values)), key=lambda i: current_values[i], reverse=descending)
                final_indices = final_indices[local_indices]

        # Apply final permutation to all columns
        for k, v in self.data.items():
            if isinstance(v, list):
                self.data[k] = [v[i] for i in final_indices]
            elif isinstance(v, np.ndarray):
                self.data[k] = v[final_indices]
            elif isinstance(v, torch.Tensor):
                indices_tensor = torch.from_numpy(final_indices).to(v.device)
                self.data[k] = v[indices_tensor]

    def map(self, func: callable, max_workers: Optional[int] = None, inplace=True) -> Union[None, 'DataDict']:
        """
        Applies a function to each row using multithreading.
        
        Args:
            func: A function that takes (index, row_dict) and returns a new row_dict.
                  The returned dictionary will be used to update the row.
                  If returns None, no update happens for that row.
            max_workers: Number of worker threads. Defaults to None (number of CPUs).
            inplace: Performs inplace operation: Defaults to True
        """
        def task(i):
            row = self[i]
            return func(i, row)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(task, range(self.length)))
            
            if inplace:
                for i, new_row in enumerate(results):
                    if new_row is not None:
                        self[i] = new_row
                return None
            
            valid_results = [r for r in results if r is not None]
            return self.from_list(valid_results)
    
    def save(self, path: str) -> None:
        """
        Saves columns to disk.
        Delegates actual saving to _save method for inheritance support.
        """
        self._save(path)

    def _save(self, path: str) -> None:
        """
        Internal implementation of save.
        - Tensors and NumPy arrays are saved as .pt
        - Lists are saved as .json
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            
        for key, col_data in self.data.items():
            file_path_pt = os.path.join(path, f"{key}.pt")
            file_path_json = os.path.join(path, f"{key}.json")
            
            if isinstance(col_data, torch.Tensor):
                torch.save(col_data, file_path_pt)
                
            elif isinstance(col_data, np.ndarray):
                torch.save(torch.from_numpy(col_data), file_path_pt)
                
            elif isinstance(col_data, list):
                with open(file_path_json, 'w') as f:
                    json.dump(col_data, f)
            else:
                try:
                    with open(file_path_json, 'w') as f:
                        json.dump(col_data, f)
                except TypeError:
                    print(f"Warning: Could not save column '{key}' of type {type(col_data)}")

    @classmethod
    def load(cls, path: str, keys: List[str]) -> 'DataDict':
        """
        Loads data from disk.
        Args:
            path: Directory containing files
            keys: List of column names to look for.
        """
        obj = cls()
        loaded_data = cls._load(path, keys)
        if loaded_data:
            obj._add_data(loaded_data)
        return obj

    @classmethod
    def _load(cls, path: str, keys: List[str]) -> Dict[str, ColumnType]:
        """
        Internal implementation of load.
        Returns a dictionary of loaded data.
        """
        if not os.path.exists(path):
             raise FileNotFoundError(f"Path not found: {path}")
             
        data_dir_files = os.listdir(path)
        loaded_data = {}
        
        for key in keys:
            if f"{key}.json" in data_dir_files:
                with open(os.path.join(path, f"{key}.json"), 'r') as f:
                    loaded_data[key] = json.load(f)
                    continue

            if f"{key}.pt" in data_dir_files:
                loaded_data[key] = torch.load(os.path.join(path, f"{key}.pt"))
                continue
        
        return loaded_data