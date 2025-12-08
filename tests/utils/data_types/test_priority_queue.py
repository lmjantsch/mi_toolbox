import pytest
from mi_toolbox.utils.data_types.priority_queue import PriorityQueue, PriorityScheme

# --- Fixtures & Data ---

@pytest.fixture
def simple_data():
    return [
        {"id": 1, "score": 10},
        {"id": 2, "score": 30},
        {"id": 3, "score": 20},
    ]

# --- Functional Tests ---

def test_simple_ascending(simple_data):
    """Test standard Min-Heap behavior (lowest value first)."""
    scheme = PriorityScheme(key="score", reversed=False)
    pq = PriorityQueue(schemes=scheme, data=simple_data)

    assert pq.pop()["score"] == 10
    assert pq.pop()["score"] == 20
    assert pq.pop()["score"] == 30

def test_simple_descending(simple_data):
    """Test Max-Heap behavior (highest value first) using reversed=True."""
    scheme = PriorityScheme(key="score", reversed=True)
    pq = PriorityQueue(schemes=scheme, data=simple_data)

    assert pq.pop()["score"] == 30
    assert pq.pop()["score"] == 20
    assert pq.pop()["score"] == 10

def test_multi_key_sorting():
    """
    Test sorting by primary key, then secondary key.
    Data:
      A: group=1, val=50
      B: group=1, val=10
      C: group=2, val=100
    Expected (Group Asc, Val Asc): B (1, 10) -> A (1, 50) -> C (2, 100)
    """
    data = [
        {"name": "A", "group": 1, "val": 50},
        {"name": "C", "group": 2, "val": 100},
        {"name": "B", "group": 1, "val": 10},
    ]
    
    schemes = [
        PriorityScheme(key="group"), # Primary: Ascending
        PriorityScheme(key="val")    # Secondary: Ascending
    ]
    pq = PriorityQueue(schemes=schemes, data=data)

    assert pq.pop()["name"] == "B"
    assert pq.pop()["name"] == "A"
    assert pq.pop()["name"] == "C"

def test_mixed_direction_sorting():
    """
    Test sorting where Primary is Ascending, but Secondary is Descending.
    Data:
      A: group=1, score=10
      B: group=1, score=90
    Expected: B (1, 90) -> A (1, 10) because for group 1, higher score wins.
    """
    data = [
        {"name": "A", "group": 1, "score": 10},
        {"name": "B", "group": 1, "score": 90},
    ]
    
    schemes = [
        PriorityScheme(key="group", reversed=False), # Ascending (1 is best)
        PriorityScheme(key="score", reversed=True)   # Descending (90 is better than 10)
    ]
    pq = PriorityQueue(schemes=schemes, data=data)

    assert pq.pop()["name"] == "B"
    assert pq.pop()["name"] == "A"

def test_custom_categorical_order():
    """Test sorting based on a specific list of strings (e.g. Low/Med/High)."""
    # We want "High" to be priority 0 (top), "Medium" priority 1, etc.
    custom_order = ["High", "Medium", "Low"]
    
    data = [
        {"task": "Sleep", "priority": "Low"},
        {"task": "Fix Bug", "priority": "High"},
        {"task": "Eat", "priority": "Medium"},
    ]
    
    scheme = PriorityScheme(key="priority", order=custom_order)
    pq = PriorityQueue(schemes=scheme, data=data)

    assert pq.pop()["task"] == "Fix Bug" # High
    assert pq.pop()["task"] == "Eat"     # Medium
    assert pq.pop()["task"] == "Sleep"   # Low

def test_stability_fifo():
    """
    Test that items with equal priority retain insertion order (FIFO).
    Crucial for predictable behavior.
    """
    data = [
        {"name": "First", "prio": 1},
        {"name": "Second", "prio": 1},
    ]
    scheme = PriorityScheme(key="prio")
    pq = PriorityQueue(schemes=scheme, data=data)

    assert pq.pop()["name"] == "First"
    assert pq.pop()["name"] == "Second"

# --- Edge Case & Error Handling Tests ---

def test_empty_queue_errors():
    """Test popping/peeking from empty queue raises IndexError."""
    scheme = PriorityScheme(key="a")
    pq = PriorityQueue(schemes=scheme)
    
    assert pq.is_empty() is True
    
    with pytest.raises(IndexError, match="pop from empty"):
        pq.pop()
        
    with pytest.raises(IndexError, match="peek from empty"):
        pq.peek()

def test_missing_key_error():
    """Test that pushing data missing the priority key raises KeyError."""
    scheme = PriorityScheme(key="required_key")
    pq = PriorityQueue(schemes=scheme)
    
    bad_row = {"other_key": 10}
    
    with pytest.raises(KeyError, match="required_key"):
        pq.push(bad_row)

def test_invalid_categorical_value():
    """Test that using a value not in the custom order list raises ValueError."""
    scheme = PriorityScheme(key="status", order=["Open", "Closed"])
    pq = PriorityQueue(schemes=scheme)
    
    bad_row = {"status": "In Progress"} # Not in ["Open", "Closed"]
    
    with pytest.raises(ValueError, match="not defined in the custom order"):
        pq.push(bad_row)

def test_duplicate_order_definition():
    """Test that defining a custom order with duplicates raises ValueError."""
    with pytest.raises(ValueError, match="duplicates"):
        PriorityScheme(key="x", order=["A", "B", "A"])

def test_init_invalid_data_type():
    """Test initializing with non-list/dict data raises TypeError."""
    scheme = PriorityScheme(key="x")
    with pytest.raises(TypeError):
        PriorityQueue(schemes=scheme, data="Not a list or dict")