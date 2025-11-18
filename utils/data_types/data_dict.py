import os
import json
from typing import Union, List, Dict
import numpy as np
import torch

COLUMN_LIKE = (list, np.ndarray, torch.Tensor)

class DataDict:

    def __init__(self, length=0):
        self.data = {}
        self.default_entry = {}
        self.length = length

    def __getitem__(self, key: Union[int, str, slice]) -> Union['DataDict', List, Dict]:
        if isinstance(key, int):
            return {k:self.data[k][key] for k in self.data}
        if isinstance(key, str):
            return self.data[key]
        if isinstance(key, slice):
            return type(self).from_dict({k:self.data[k][key] for k in self.data})
        if isinstance(key, COLUMN_LIKE):
            if all(isinstance(el, str) for el in key):
                return type(self).from_dict({key_el: self.data[key_el] for key_el in key})
            if all(isinstance(el, int) for el in key):
                return type(self).from_dict({k:[self.data[k][key_el] for key_el in key] for k in self.default_entry})
            
            raise TypeError(f"Unsupported key type: {type(key)}[any]")
        raise TypeError(f"Unsupported key type: {type(key)}")
    
    def __setitem__(self, key: Union[int, str, slice], value: any) -> None:
        if isinstance(key, int):
            if not isinstance(value, dict):
                raise TypeError('Expected dictionary for row editing')
            if not set(value).issubset(self.default_entry):
                raise TypeError(f'Unexpected Keys: {list(set(value) - set(self.default_entry))}')
            for value_k, value_v in value.items():
                self.data[value_k][key] = value_v
            return
        if isinstance(key, str):
            if not isinstance(value, COLUMN_LIKE):
                raise TypeError("Expected list for column editing")
            if len(value) != self.length:
                raise ValueError(f"Length mismatch: The new column has {len(value)} entries, but {self.length} are expected.")
            self.data[key] = value
            return
        if isinstance(key, slice):
            if not isinstance(value, dict):
                raise TypeError('Expected dictionary of lists for row slice editing')
            if not set(value).issubset(self.default_entry):
                raise TypeError(f'Unexpected Keys: {list(set(value) - set(self.default_entry))}')
            slice_len = key.stop - key.start
            for value_k, value_v in value.items():
                if not isinstance(value_v, COLUMN_LIKE):
                    raise TypeError('Expected dictionary of lists for row slice editing')
                if slice_len != len(value_v):
                    raise ValueError(f"Length mismatch: There are {len(value_v)} entries for column {value_k}, but {slice_len} are expected.")
            for value_k, value_v in value.items():
                self.data[value_k][key] = value_v
            return
        if isinstance(key, COLUMN_LIKE):
            if all(isinstance(el, str) for el in key):
                if not isinstance(value, COLUMN_LIKE):
                    raise TypeError('Expected list of lists for multi column editing')
                if len(key) != len(value):
                    raise TypeError(f'Expected {len(key)} new columns, but {len(value)} are provided.')
                for key, value in zip(key, value):
                    self[key] = value
                return
            if all(isinstance(el, int) for el in key):
                if not isinstance(value, COLUMN_LIKE):
                    raise TypeError('Expected list of dicts for multi row editing')
                if len(key) != len(value):
                    raise TypeError(f'Expected {len(key)} new rows, but {len(value)} are provided.')
                for key_k, value_dict in zip(key, value):
                    self[key_k] = value_dict
                return

            raise TypeError(f"Unsupported key type: {type(key)}[any]")
        raise TypeError(f"Unsupported key type: {type(key)}")

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.data.keys())

    def __repr__(self):
        return f"{self.__class__.__name__}(entries={self.length}, keys={self.default_entry.keys()})"
    
    def append(self, data: Dict):
        if not isinstance(data, dict):
            raise TypeError(f"Dictionary expected but found {type(data)}.")
        self._add_data({k: [v] for k, v in data.items()})
    
    def extend(self, data: Dict[str, List]):
        if not isinstance(data, dict):
            raise TypeError(f"Dictionary of list expected but found {type(data)}.")
        if not all([isinstance(v, COLUMN_LIKE) for v in data.values()]):
            raise TypeError(f"All values must be column like (list, ndarray, tensor)")
        self._add_data(data)

    def attach(self, key: str, data: List, force: bool=False):
        if not force and key in self.data:
            raise ValueError(f"The key '{key}' already exists.")
        if len(data) != self.length:
            raise ValueError(f"Length mismatch: The new column has {len(data)} entries, but {self.length} are expected.")
        self.default_entry[key] = type(data[0])()
        self.data[key] = data
    
    @ classmethod
    def from_list(cls, data: List[Dict], **kwargs):
        obj = cls(**kwargs)
        if not data:
            return obj
        obj._set_default_entry(data[0])
        obj.data = {k:[] for k in obj.default_entry.keys()}
        for sample in data:
            if not set(sample).issubset(obj.default_entry):
                raise TypeError(f'Unexpected Keys: {list(set(sample) - set(obj.default_entry))}')
            for k, default_value in obj.default_entry.items():
                obj.data[k].append(sample.get(k, default_value))
        obj.length = len(obj.data[k])
        return obj
    
    @ classmethod
    def from_dict(cls, data: Dict[str, List], **kwargs):
        obj = cls(**kwargs)
        obj._set_default_entry({k:v[0] for k,v in data.items()})
        for k in obj.default_entry.keys():
            if not obj.length:
                obj.length = len(data[k])
            if obj.length != len(data[k]):
                raise ValueError(f"The data length of {k} ({len(data[k])}) does not match the object length ({len(data[k])})")
            obj.data[k] = data[k]
        return obj
    
    def _add_data(self, data: Dict[str, List]):
        if not self.default_entry:
            self._set_default_entry({k:v[0] for k,v in data.items()})
            self.data = {k:[] for k in self.default_entry}
        if not set(data).issubset(self.default_entry):
            raise TypeError(f'Unexpected Keys: {list(set(data) - set(self.default_entry))}')
        num_new_rows = next(len(v) for v in data.values())
        for k, v in data.items():
            if num_new_rows != len(v):
                raise ValueError(f"The data length of {k} ({len(v)}) does not match the object length ({num_new_rows})")
        for k, default_v in self.default_entry.items():
            if k not in data:
                self.data[k].extend([default_v] * num_new_rows)
            else:
                self.data[k].extend(data[k])
        self.length += num_new_rows
    
    def _set_default_entry(self, item:Dict):
        self.default_entry = {k: type(v)() if v != None else None for k, v in item.items()}

    def keys(self):
        return self.default_entry.keys()
    
    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()
    
    def to_list(self):
        keys = list(self.keys())
        cols = [self.data[k] for k in keys]
        return [dict(zip(keys, row)) for row in zip(*cols)]
        
    def to_dict(self):
        return self.data
    
    def sort(self, by: Union[List, str], descending=False):
        if isinstance(by, str):
            by = [by]

        for by_key in reversed(by):
            if by_key not in self.default_entry:
                raise KeyError(f"The sort key '{by_key}' does not exist in {self.default_entry.keys()}.")
            sorting_idx = sorted(list(range(self.length)), key=lambda i: self.data[by_key][i], reverse=descending)
            for key in self.default_entry:
                self.data[key] = [self.data[key][i] for i in sorting_idx]

    def map(self, fn:callable) -> None:
        for i in range(self.length):
            row = self[i]
            self[i] = fn(i, row)

    @ classmethod
    def load(cls, path: str, keys: List[str], **kwargs) -> 'DataDict':
        obj = cls(**kwargs)
        data_dir_files = os.listdir(path)
        for key in keys:

            if f"{key}.json" in data_dir_files:
                file_path = os.path.join(path, f"{key}.json")
                with open(file_path, 'r') as f:
                    data = json.load(f)

            if f"{key}.safetensors" in data_dir_files:
                file_path  = os.path.join(path, f"{key}.safetensors")
                data = torch.load(file_path)
            
            if not obj.length:
                obj.length = len(data)

            obj.attach(key, data)
        
        return obj