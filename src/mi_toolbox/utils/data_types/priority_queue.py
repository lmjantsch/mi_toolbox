import heapq
from typing import Optional, List, Union, Tuple, Dict, Any

class PriorityScheme:
    """
    Defines how a specific key in a dictionary should be prioritized.
    Supports standard ordering, reversed ordering, and arbitrary custom ordering (categorical).
    """

    def __init__(self, key: str, order: Optional[List[Any]] = None, reversed: bool = False):
        """
        Initialize the priority scheme.

        Args:
            key (str): The dictionary key to inspect.
            order (Optional[List[Any]]): A specific list defining the order of values (e.g., ["low", "med", "high"]).
                                         If None, standard comparison operators (<, >) are used on the values.
            reversed (bool): If True, reverses the sort order (Descedning). 
                             Default is False (Ascending/Min-Heap style).
        
        Raises:
            ValueError: If 'order' list contains duplicate values.
        """
        if order and len(order) != len(set(order)):
            raise ValueError(f"Order list for key '{key}' contains duplicates, which prevents unambiguous sorting.")

        self.key = key
        # Convert list to dict for O(1) lookup speed
        self.order_map = {v: i for i, v in enumerate(order)} if order else None
        self.reversed = reversed

    def extract_value(self, data: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Extracts the value from the data dict and prepares it for comparison.

        Args:
            data (Dict[str, Any]): The data row.

        Returns:
            Tuple[Any, bool]: A tuple containing the comparable value and the reversed flag.

        Raises:
            TypeError: If data is not a dictionary.
            KeyError: If the required key is missing from data.
            ValueError: If using a custom order and the value is not found in the order list.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected data to be a dict, got {type(data).__name__}")

        if self.key not in data:
            raise KeyError(f"Priority key '{self.key}' is missing from the provided data.")
        
        raw_val = data[self.key]
        
        # If a custom categorical order is defined, map the raw value to its index (int)
        if self.order_map is not None:
            try:
                val = self.order_map[raw_val]
            except KeyError:
                raise ValueError(f"Value '{raw_val}' for key '{self.key}' is not defined in the custom order list.")
        else:
            val = raw_val
                
        return (val, self.reversed)


class PriorityObject:
    """
    Internal wrapper class to handle multi-key comparisons with mixed directions (asc/desc).
    """
    
    def __init__(self, values_with_directions: List[Tuple[Any, bool]]):
        """
        Args:
            values_with_directions: A list of (value, is_reversed) tuples.
        """
        self.items = values_with_directions

    def __lt__(self, other: Any) -> bool:
        """
        Compare this object with another. 
        Iterates through priority values; the first non-equal value determines the result.
        """
        if not isinstance(other, PriorityObject):
            raise TypeError(f"Cannot compare PriorityObject with {type(other).__name__}")

        for (val_a, is_rev), (val_b, _) in zip(self.items, other.items):
            if val_a == val_b:
                continue

            if is_rev:
                return val_a > val_b
            return val_a < val_b
            
        # If we reach here, all values are equal
        return False

    def __eq__(self, other: Any) -> bool:
        """Checks if all underlying priority values are equal."""
        if not isinstance(other, PriorityObject):
            return False
        return all(a[0] == b[0] for a, b in zip(self.items, other.items))

    def __str__(self):
        return f"PriorityObject({[x[0] for x in self.items]})"


class PriorityQueue:
    """
    A generic Priority Queue supporting multi-column sorting and mixed sort directions.
    Backed by Python's heapq (Min-Heap).
    """

    def __init__(self, schemes: Union[PriorityScheme, List[PriorityScheme]], data: Optional[Union[Dict, List[Dict]]] = None):
        """
        Args:
            schemes (Union[PriorityScheme, List[PriorityScheme]]): One or more PriorityScheme objects defining sort order.
            data (Optional[Union[Dict, List[Dict]]]): Initial data to populate the queue.

        Raises:
            ValueError: If schemes contain objects that are not PriorityScheme instances.
        """
        if isinstance(schemes, PriorityScheme):
            schemes = [schemes]
            
        if not all(isinstance(s, PriorityScheme) for s in schemes):
            raise ValueError("All items in 'schemes' must be instances of PriorityScheme.")

        self.schemes = schemes
        # Heap stores tuples: (PriorityObject, unique_id, actual_data)
        self._heap = [] 
        self._counter = 0 # Tie-breaker to ensure stability and avoid dict comparisons
        
        if data is not None:
            if isinstance(data, list):
                for row in data:
                    self.push(row)
            elif isinstance(data, dict):
                self.push(data)
            else:
                raise TypeError(f"Initial data must be a dict or list of dicts, got {type(data).__name__}")

    def _create_priority(self, row: Dict) -> PriorityObject:
        """Helper to generate the comparison object for a given row."""
        values = [scheme.extract_value(row) for scheme in self.schemes]
        return PriorityObject(values)

    def push(self, row: Dict) -> None:
        """
        Push a new item onto the queue.

        Args:
            row (Dict): The data dictionary to add. Must contain keys defined in schemes.
        """
        priority = self._create_priority(row)
        
        # We push: (Priority wrapper, insert order, actual data)
        # 'insert order' (_counter) acts as a tie-breaker. If priorities are equal,
        # the item inserted first is popped first (FIFO for ties).
        heapq.heappush(self._heap, (priority, self._counter, row))
        self._counter += 1

    def pop(self) -> Dict:
        """
        Remove and return the highest priority item (the 'smallest' in Min-Heap terms).

        Returns:
            Dict: The data dictionary.

        Raises:
            IndexError: If the queue is empty.
        """
        if not self._heap:
            raise IndexError("pop from empty PriorityQueue")
        
        _, _, row = heapq.heappop(self._heap)
        return row

    def peek(self) -> Dict:
        """
        Return the highest priority item without removing it.

        Returns:
            Dict: The data dictionary.

        Raises:
            IndexError: If the queue is empty.
        """
        if not self._heap:
            raise IndexError("peek from empty PriorityQueue")
        
        # Index 0 is the smallest item (root of heap)
        # Tuple index 2 is the actual data
        return self._heap[0][2]
    
    def is_empty(self) -> bool:
        """Returns True if the queue has no items."""
        return len(self._heap) == 0

    def __len__(self) -> int:
        """Returns the number of items in the queue."""
        return len(self._heap)