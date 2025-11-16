from typing import Union, List, Dict

class DataDict:

    def __init__(self):
        self.data = {}
        self.default_entry = {}
        self.length = 0

    def __getitem__(self, key: Union[int, str, slice]) -> Union['DataDict', List, Dict]:
        if isinstance(key, int):
            return {k:self.data[k][key] for k in self.data}
        if isinstance(key, str):
            return self.data[key]
        if isinstance(key, slice):
            return type(self).from_dict({k:self.data[k][key] for k in self.data})
        if isinstance(key, list):
            if all(isinstance(el, str) for el in key):
                return type(self).from_dict({key_el: self.data[key_el] for key_el in key})
            if all(isinstance(el, int) for el in key):
                return type(self).from_dict({k:[self.data[k][key_el] for key_el in key] for k in self.default_entry})
        raise TypeError(f"Unsupported key type: {type(key)}")

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.data.keys())

    def __repr__(self):
        return f"{self.__class__.__name__}(entries={self.length}, keys={self.default_entry.keys()})"
    
    def append(self, data: Dict):
        self._add_data({k: [v] for k, v in data.items()})
    
    def extend(self, data: Dict[str, List]):
        self._add_data(data)

    def attach(self, key: str, data: List, force: bool=False):
        if not force and key in self.data:
            raise ValueError(f"The key '{key}' already exists.")
        if len(data) != self.length:
            raise ValueError(f"Length mismatch: The new column has {len(data)} entries, but {self.length} are expected.")
        self.default_entry[key] = type(data[0])()
        self.data[key] = data
    
    @ classmethod
    def from_list(cls, data: List[Dict]):
        obj = cls()
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
    def from_dict(cls, data: Dict[str, List]):
        obj = cls()
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
                raise ValueError(f"The data length of {k} ({len(v)}) does not match the object length ({len(v)})")
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
            sorting_idx = sorted(list(range(self.length)), key=lambda i: self.data[by_key][i])
            for key in self.default_entry:
                self.data[key] = [self.data[key][i] for i in sorting_idx]