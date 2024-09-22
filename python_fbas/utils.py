from pysat.formula import Formula

def to_cnf(fmlas : list[Formula]) -> list:
    """Convert a list of formulas to CNF."""
    # NOTE: iterating through f first triggers clausification
    return [c for f in fmlas for c in f]

def clause_of_pseudo_atom(a: Formula) -> list:
    """a must be of the form Neg(Atom(...)); returns the corresponding clause"""
    a.clausify()
    return a.clauses[0]

def fixpoint(f, x):
    """Iterate f starting from x until a fixed point is reached."""
    while True:
        y = f(x)
        if x == y:
            return x
        x = y