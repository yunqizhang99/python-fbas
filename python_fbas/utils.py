def fixpoint(f, x):
    """Iterate f starting from x until a fixed point is reached."""
    while True:
        y = f(x)
        if x == y:
            return x
        x = y