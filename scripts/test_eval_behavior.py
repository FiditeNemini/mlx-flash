import mlx.core as mx


class MockCache:
    def __init__(self, arr):
        self.arr = arr

def test_eval():
    a = mx.array([1.0])
    b = mx.array([2.0])
    c = a + b
    
    cache = MockCache(c)
    print(f"Before eval, c is realized: {c.is_realized}")
    
    # Try eval on object
    mx.eval(cache)
    print(f"After mx.eval(cache), c is realized: {c.is_realized}")
    
    # Try eval on nested
    d = mx.array([3.0])
    e = c + d
    nested = {"k": [e]}
    mx.eval(nested)
    print(f"After mx.eval(nested), e is realized: {e.is_realized}")

if __name__ == "__main__":
    test_eval()
