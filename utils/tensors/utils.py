from typing import Tuple

def is_broadcastable(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> bool:
    
    s1 = list(reversed(shape1))
    s2 = list(reversed(shape2))
    
    max_dims = max(len(s1), len(s2))
    
    while len(s1) < max_dims:
        s1.append(1)
    while len(s2) < max_dims:
        s2.append(1)
        
    for dim1, dim2 in zip(s1, s2):
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            return False
            
    return True